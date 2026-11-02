"""Sequential five-fold inference for the final RSNA DINOv2 model.

This module consumes the exact per-study arrays saved by Phase 2A and the
one-fold strict checkpoint loader from Phase 2B. It follows the Kaggle final
inference rule: sigmoid each fold's logits independently, then average the
five probability vectors (not logits).

Only one fold model is kept in memory at a time, which is friendlier to local
CPU/RAM constraints than the original Kaggle notebook's GPU-resident list.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..models.dinov2_bundle import load_dinov2_fold
from ..models.dinov2_five_role import FIVE_ROLE_ORDER, TARGETS


EXPECTED_IMAGES_SHAPE = (5, 10, 3, 224, 224)
EXPECTED_MASK_SHAPE = (5, 10)


class DinoInferenceError(RuntimeError):
    """Raised when DINOv2 study inference cannot safely proceed."""


@dataclass
class DinoV2EnsembleResult:
    """Five-fold per-study probabilities and execution metadata."""

    study_uid: str
    status: str
    fold_probabilities: np.ndarray  # float32 [5, 12]
    ensemble_probabilities: np.ndarray  # float32 [12]
    fold_info: list[dict[str, Any]]
    device: str
    encoder_chunk_size: int
    amp_enabled: bool
    n_valid_windows: int
    n_available_roles: int
    elapsed_seconds: float
    aggregation: str = "mean of per-fold sigmoid probabilities"

    @property
    def fold_std(self) -> np.ndarray:
        """Across-fold standard deviation; descriptive, not calibrated uncertainty."""
        return self.fold_probabilities.std(axis=0, ddof=0).astype(np.float32)


FoldLoader = Callable[..., tuple[torch.nn.Module, dict[str, Any]]]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise DinoInferenceError("CUDA was requested but torch.cuda.is_available() is False.")
    return result


def validate_dinov2_inputs(
    images: np.ndarray,
    window_valid: np.ndarray,
    window_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the Phase 2A array contract and return normalized dtypes.

    Image values must remain uint8. Scaling and ImageNet normalization happen
    inside ``FiveRoleDINOv2_2p5D.forward`` exactly once.
    """
    images_np = np.asarray(images)
    valid_np = np.asarray(window_valid)
    positions_np = np.asarray(window_positions)

    if images_np.shape != EXPECTED_IMAGES_SHAPE:
        raise DinoInferenceError(
            f"images must have shape {EXPECTED_IMAGES_SHAPE}; got {images_np.shape}."
        )
    if images_np.dtype != np.uint8:
        raise DinoInferenceError(
            f"images must remain uint8 (no pre-normalization); got {images_np.dtype}."
        )
    if valid_np.shape != EXPECTED_MASK_SHAPE:
        raise DinoInferenceError(
            f"window_valid must have shape {EXPECTED_MASK_SHAPE}; got {valid_np.shape}."
        )
    if positions_np.shape != EXPECTED_MASK_SHAPE:
        raise DinoInferenceError(
            f"window_positions must have shape {EXPECTED_MASK_SHAPE}; got {positions_np.shape}."
        )
    if not np.isfinite(positions_np).all():
        raise DinoInferenceError("window_positions contains NaN or Inf.")

    valid_bool = valid_np.astype(bool, copy=False)
    positions_f32 = positions_np.astype(np.float32, copy=False)
    if valid_bool.any():
        valid_positions = positions_f32[valid_bool]
        if np.any(valid_positions < -1e-5) or np.any(valid_positions > 1.00001):
            raise DinoInferenceError("Valid window_positions must be normalized to [0, 1].")

    return images_np, valid_bool, positions_f32


def run_dinov2_five_fold_inference(
    *,
    images: np.ndarray,
    window_valid: np.ndarray,
    window_positions: np.ndarray,
    bundle_root: str | Path,
    study_uid: str = "unknown",
    device: str | torch.device | None = None,
    encoder_chunk_size: int | None = None,
    use_amp: bool = True,
    fold_loader: FoldLoader = load_dinov2_fold,
    verbose: bool = True,
) -> DinoV2EnsembleResult:
    """Run all five folds sequentially and average their target probabilities.

    Kaggle parity:
      1. Call each fold model on [1, 5, 10, 3, 224, 224].
      2. Apply sigmoid to that fold's logits.
      3. Average the five probability vectors along the fold dimension.

    Resource behavior:
      - Load one checkpoint/model at a time.
      - Release the model and collect garbage after every fold.
      - Use smaller encoder chunks on CPU (default 2) than Kaggle GPU (8).

    When no valid window exists, match the final Kaggle inference fallback:
    every fold probability and the resulting mean are 0.5. No checkpoint is
    loaded in that case; the result status makes the fallback explicit.
    """
    images_np, valid_np, positions_np = validate_dinov2_inputs(
        images, window_valid, window_positions
    )
    device_obj = _resolve_device(device)
    if encoder_chunk_size is None:
        encoder_chunk_size = 8 if device_obj.type == "cuda" else 2
    if int(encoder_chunk_size) < 1:
        raise ValueError("encoder_chunk_size must be >= 1.")
    encoder_chunk_size = int(encoder_chunk_size)
    amp_enabled = bool(use_amp and device_obj.type == "cuda")

    n_valid_windows = int(valid_np.sum())
    n_available_roles = int(valid_np.any(axis=1).sum())
    started_all = time.perf_counter()

    # Kaggle's fold_probabilities array starts at 0.5; a study with no usable
    # image windows keeps that neutral value and does not call any fold model.
    if n_valid_windows == 0:
        neutral = np.full((5, len(TARGETS)), 0.5, dtype=np.float32)
        return DinoV2EnsembleResult(
            study_uid=str(study_uid),
            status="all_roles_unusable_fallback_0.5",
            fold_probabilities=neutral,
            ensemble_probabilities=neutral.mean(axis=0).astype(np.float32),
            fold_info=[],
            device=str(device_obj),
            encoder_chunk_size=encoder_chunk_size,
            amp_enabled=amp_enabled,
            n_valid_windows=0,
            n_available_roles=0,
            elapsed_seconds=time.perf_counter() - started_all,
        )

    # Allocate the full study input once. The model itself encodes only valid
    # windows, as implemented in the exact FiveRoleDINOv2_2p5D.forward().
    images_t = torch.from_numpy(images_np[None]).to(device_obj)
    valid_t = torch.from_numpy(valid_np[None]).to(device_obj)
    positions_t = torch.from_numpy(positions_np[None]).to(device_obj)

    fold_probabilities = np.empty((5, len(TARGETS)), dtype=np.float32)
    fold_info_all: list[dict[str, Any]] = []

    for fold in range(1, 6):
        model: torch.nn.Module | None = None
        fold_started = time.perf_counter()
        try:
            model, model_info = fold_loader(
                bundle_root=bundle_root,
                fold=fold,
                device=device_obj,
                encoder_chunk_size=encoder_chunk_size,
            )
            model.eval()

            with torch.inference_mode():
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        logits = model(images_t, valid_t, positions_t)
                else:
                    logits = model(images_t, valid_t, positions_t)

                if tuple(logits.shape) != (1, len(TARGETS)):
                    raise DinoInferenceError(
                        f"Fold {fold}: expected logits shape (1, {len(TARGETS)}), "
                        f"received {tuple(logits.shape)}."
                    )
                probabilities = torch.sigmoid(logits.float()).squeeze(0).cpu().numpy()

            probabilities = np.asarray(probabilities, dtype=np.float32)
            if probabilities.shape != (len(TARGETS),):
                raise DinoInferenceError(
                    f"Fold {fold}: expected {len(TARGETS)} probabilities, "
                    f"received {probabilities.shape}."
                )
            if not np.isfinite(probabilities).all():
                raise DinoInferenceError(f"Fold {fold}: probabilities contain NaN/Inf.")
            if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
                raise DinoInferenceError(f"Fold {fold}: probabilities fall outside [0, 1].")

            fold_probabilities[fold - 1] = probabilities
            fold_elapsed = time.perf_counter() - fold_started
            fold_info = dict(model_info)
            fold_info.update(
                {
                    "fold": fold,
                    "strict_load": bool(model_info.get("strict_load", False)),
                    "inference_status": "ok",
                    "valid_windows_encoded": n_valid_windows,
                    "inference_seconds": float(fold_elapsed),
                    "probability_min": float(probabilities.min()),
                    "probability_max": float(probabilities.max()),
                }
            )
            fold_info_all.append(fold_info)
            if verbose:
                print(
                    f"Fold {fold}/5: strict_load={fold_info['strict_load']} | "
                    f"encoded_windows={n_valid_windows}/50 | "
                    f"range=[{probabilities.min():.4f}, {probabilities.max():.4f}] | "
                    f"elapsed={fold_elapsed:.1f}s",
                    flush=True,
                )
        except Exception as exc:
            raise DinoInferenceError(
                f"DINOv2 fold {fold} failed. No partial five-fold ensemble was returned. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            if model is not None:
                del model
            gc.collect()
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

    # Important: average probabilities, not logits, matching Kaggle Cell 33.
    ensemble_probabilities = fold_probabilities.mean(axis=0).astype(np.float32)
    if ensemble_probabilities.shape != (len(TARGETS),):
        raise DinoInferenceError(
            f"Unexpected ensemble probability shape: {ensemble_probabilities.shape}."
        )
    if not np.isfinite(ensemble_probabilities).all():
        raise DinoInferenceError("Five-fold mean contains NaN/Inf.")

    elapsed_total = time.perf_counter() - started_all
    if verbose:
        print(
            "Five-fold inference complete | "
            f"study={study_uid} | valid_windows={n_valid_windows}/50 | "
            f"available_roles={n_available_roles}/5 | elapsed={elapsed_total:.1f}s",
            flush=True,
        )

    # Release only the input tensors after all folds have consumed them.
    del images_t, valid_t, positions_t
    gc.collect()
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()

    return DinoV2EnsembleResult(
        study_uid=str(study_uid),
        status="ok",
        fold_probabilities=fold_probabilities,
        ensemble_probabilities=ensemble_probabilities,
        fold_info=fold_info_all,
        device=str(device_obj),
        encoder_chunk_size=encoder_chunk_size,
        amp_enabled=amp_enabled,
        n_valid_windows=n_valid_windows,
        n_available_roles=n_available_roles,
        elapsed_seconds=float(elapsed_total),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return value


def save_dinov2_inference_result(
    result: DinoV2EnsembleResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save per-fold probabilities, the five-fold mean and an audit report."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    fold_csv = output / "dinov2_fold_probabilities.csv"
    with fold_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", *TARGETS])
        writer.writeheader()
        for fold_index, probabilities in enumerate(result.fold_probabilities, start=1):
            row = {"fold": fold_index}
            row.update({target: float(probabilities[i]) for i, target in enumerate(TARGETS)})
            writer.writerow(row)

    prediction_csv = output / "dinov2_ensemble_prediction.csv"
    with prediction_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["StudyInstanceUID", *TARGETS])
        writer.writeheader()
        row = {"StudyInstanceUID": result.study_uid}
        row.update(
            {
                target: float(result.ensemble_probabilities[i])
                for i, target in enumerate(TARGETS)
            }
        )
        writer.writerow(row)

    npz_path = output / "dinov2_five_fold_predictions.npz"
    np.savez_compressed(
        npz_path,
        fold_probabilities=result.fold_probabilities,
        ensemble_probabilities=result.ensemble_probabilities,
        fold_std=result.fold_std,
        targets=np.asarray(TARGETS),
        role_order=np.asarray(FIVE_ROLE_ORDER),
        study_uid=np.asarray(result.study_uid),
        status=np.asarray(result.status),
    )

    report_path = output / "dinov2_five_fold_inference_report.json"
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_notebook": "08_model_ensemble_kaggle_dataset_ready.ipynb",
        "source_cells": {
            "architecture_and_forward": 29,
            "fold_checkpoint_loading": 31,
            "study_inference_and_probability_mean": 33,
        },
        "study_instance_uid": result.study_uid,
        "status": result.status,
        "device": result.device,
        "encoder_chunk_size": result.encoder_chunk_size,
        "amp_enabled": result.amp_enabled,
        "valid_windows": result.n_valid_windows,
        "available_roles": result.n_available_roles,
        "fold_info": result.fold_info,
        "aggregation": result.aggregation,
        "ensemble_probabilities": {
            target: float(result.ensemble_probabilities[i])
            for i, target in enumerate(TARGETS)
        },
        "fold_std": {
            target: float(result.fold_std[i]) for i, target in enumerate(TARGETS)
        },
        "elapsed_seconds": result.elapsed_seconds,
        "important_note": (
            "Software demonstration on one gold_validation study. This is not an "
            "independent performance evaluation and must not be presented as a new "
            "validation score. Fold standard deviation is descriptive only, not a "
            "calibrated uncertainty estimate."
        ),
    }
    report_path.write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "fold_probabilities_csv": fold_csv,
        "ensemble_prediction_csv": prediction_csv,
        "predictions_npz": npz_path,
        "report_json": report_path,
    }
