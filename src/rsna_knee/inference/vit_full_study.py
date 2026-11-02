"""Full-study inference for the final RSNA ViT-B/16 model.

The execution contract follows the final Kaggle inference notebook:
    logits = model(images, valid, plane_ids, fluid, fat, positions)
    probabilities = sigmoid(logits.float())

The model itself performs uint8 -> [0, 1], grayscale-to-RGB replication,
ImageNet normalization, valid-slice chunking, slice encoding and study pooling.
This module must not normalize images a second time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from rsna_knee.models.vit_study import TARGETS

EXPECTED_SLICES = 36
EXPECTED_IMAGE_SIZE = 224
EXPECTED_PLANE_ORDER = ("Sagittal", "Coronal", "Axial")
REQUIRED_ARRAYS = (
    "images",
    "valid",
    "plane_ids",
    "fluid",
    "fat",
    "positions",
)


class ViTInferenceError(RuntimeError):
    """Raised when full-study ViT inference cannot be completed safely."""


@dataclass(frozen=True)
class ViTPrediction:
    study_uid: str
    logits: np.ndarray
    probabilities: np.ndarray
    n_valid_slices: int
    valid_slices_by_plane: dict[str, int]
    device: str


def validate_vit_inputs(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate and return the fixed-length one-study Phase 2D cache arrays."""
    missing = [name for name in REQUIRED_ARRAYS if name not in payload]
    if missing:
        raise ViTInferenceError(f"ViT preprocessing cache is missing arrays: {missing}")

    arrays = {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}
    images = arrays["images"]
    if images.shape != (EXPECTED_SLICES, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE):
        raise ViTInferenceError(
            "images must have shape (36, 224, 224); "
            f"received {images.shape}."
        )
    if images.dtype != np.uint8:
        raise ViTInferenceError(f"images must be uint8, received {images.dtype}.")

    for name in ("valid", "plane_ids", "fluid", "fat", "positions"):
        if arrays[name].shape != (EXPECTED_SLICES,):
            raise ViTInferenceError(
                f"{name} must have shape (36,), received {arrays[name].shape}."
            )

    valid = arrays["valid"].astype(bool)
    if not valid.any():
        raise ViTInferenceError(
            "The study has zero valid slices. The model cannot infer this study; "
            "do not replace this failure with fabricated probabilities."
        )

    plane_ids = arrays["plane_ids"].astype(np.int64, copy=False)
    valid_plane_ids = set(np.unique(plane_ids[valid]).tolist())
    if not valid_plane_ids.issubset({0, 1, 2}):
        raise ViTInferenceError(
            f"Valid slices contain unknown plane IDs: {sorted(valid_plane_ids)}."
        )

    positions = arrays["positions"].astype(np.float32, copy=False)
    if not np.isfinite(positions).all():
        raise ViTInferenceError("positions contains NaN or Inf values.")
    if np.any((positions[valid] < -1e-6) | (positions[valid] > 1.0 + 1e-6)):
        raise ViTInferenceError("Valid slice positions must be normalized to [0, 1].")

    for name in ("fluid", "fat"):
        values = arrays[name].astype(np.int64, copy=False)
        if not set(np.unique(values[valid]).tolist()).issubset({-1, 0, 1}):
            raise ViTInferenceError(
                f"{name} metadata must use -1/0/1 on valid slices."
            )

    return arrays


def predict_vit_full_study(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    study_uid: str,
    device: str | torch.device,
) -> ViTPrediction:
    """Run one complete 36-slot study through the restored ViT model.

    All valid slices are passed to the model. Its configured
    ``slice_chunk_size`` bounds the encoder's activation memory. No valid slice
    is removed for this full-study inference.
    """
    arrays = validate_vit_inputs(payload)
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    images_t = torch.from_numpy(np.array(arrays["images"], copy=True)[None]).to(device_obj)
    valid_t = torch.from_numpy(
        np.array(arrays["valid"], copy=True).astype(np.uint8)[None]
    ).to(device_obj)
    plane_t = torch.from_numpy(
        np.array(arrays["plane_ids"], copy=True).astype(np.int64)[None]
    ).to(device_obj)
    fluid_t = torch.from_numpy(
        np.array(arrays["fluid"], copy=True).astype(np.float32)[None]
    ).to(device_obj)
    fat_t = torch.from_numpy(
        np.array(arrays["fat"], copy=True).astype(np.float32)[None]
    ).to(device_obj)
    positions_t = torch.from_numpy(
        np.array(arrays["positions"], copy=True).astype(np.float32)[None]
    ).to(device_obj)

    try:
        with torch.inference_mode():
            logits_t = model(
                images_t,
                valid_t,
                plane_t,
                fluid_t,
                fat_t,
                positions_t,
            )
            probabilities_t = torch.sigmoid(logits_t.float())
    except Exception as exc:
        raise ViTInferenceError(
            f"Full-study ViT forward failed for study {study_uid}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    expected_shape = (1, len(TARGETS))
    if tuple(logits_t.shape) != expected_shape:
        raise ViTInferenceError(
            f"Model logits must have shape {expected_shape}, got {tuple(logits_t.shape)}."
        )
    if tuple(probabilities_t.shape) != expected_shape:
        raise ViTInferenceError(
            f"Model probabilities must have shape {expected_shape}, "
            f"got {tuple(probabilities_t.shape)}."
        )

    logits = logits_t.detach().float().cpu().numpy()[0].astype(np.float32)
    probabilities = probabilities_t.detach().float().cpu().numpy()[0].astype(np.float32)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ViTInferenceError("Model output contains NaN or Inf values.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ViTInferenceError("Sigmoid probabilities must be within [0, 1].")

    valid = arrays["valid"].astype(bool)
    plane_ids = arrays["plane_ids"].astype(np.int64, copy=False)
    by_plane = {
        plane: int(np.sum(valid & (plane_ids == plane_id)))
        for plane_id, plane in enumerate(EXPECTED_PLANE_ORDER)
    }

    return ViTPrediction(
        study_uid=str(study_uid),
        logits=logits,
        probabilities=probabilities,
        n_valid_slices=int(valid.sum()),
        valid_slices_by_plane=by_plane,
        device=str(device_obj),
    )


def save_vit_full_study_outputs(
    prediction: ViTPrediction,
    output_dir: str | Path,
    *,
    model_info: Mapping[str, Any],
    restore_seconds: float,
    inference_seconds: float,
    cache_path: str | Path,
) -> dict[str, Path]:
    """Write per-target CSV, numeric NPZ and a JSON provenance report."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "vit_full_study_predictions.csv"
    npz_path = output / "vit_full_study_prediction.npz"
    report_path = output / "vit_full_study_inference_report.json"

    import pandas as pd

    pd.DataFrame(
        {
            "target": list(TARGETS),
            "logit": prediction.logits,
            "probability": prediction.probabilities,
        }
    ).to_csv(csv_path, index=False, float_format="%.10g")

    np.savez_compressed(
        npz_path,
        study_uid=np.asarray(prediction.study_uid),
        targets=np.asarray(TARGETS),
        logits=prediction.logits.astype(np.float32),
        probabilities=prediction.probabilities.astype(np.float32),
    )

    report = {
        "status": "success",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_instance_uid": prediction.study_uid,
        "source_preprocessed_cache": str(Path(cache_path).resolve()),
        "architecture": model_info.get("architecture"),
        "checkpoint_name": model_info.get("checkpoint_name"),
        "artifact": model_info.get("artifact"),
        "artifact_member": model_info.get("member"),
        "best_epoch": model_info.get("best_epoch"),
        "best_val_loss": model_info.get("best_val_loss"),
        "strict_load": bool(model_info.get("strict_load", False)),
        "parameter_count": int(model_info.get("parameter_count", 0)),
        "device": prediction.device,
        "runtime_config": model_info.get("run_config", {}),
        "input": {
            "total_slots": EXPECTED_SLICES,
            "valid_slices": prediction.n_valid_slices,
            "valid_slices_by_plane": prediction.valid_slices_by_plane,
            "tensor_shape": [1, EXPECTED_SLICES, EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE],
        },
        "output": {
            "logits_shape": [len(TARGETS)],
            "probabilities_shape": [len(TARGETS)],
            "all_finite": bool(np.isfinite(prediction.probabilities).all()),
            "probability_min": float(prediction.probabilities.min()),
            "probability_max": float(prediction.probabilities.max()),
            "targets": list(TARGETS),
        },
        "timing_seconds": {
            "checkpoint_restore": float(restore_seconds),
            "full_study_inference": float(inference_seconds),
        },
        "inference_contract": {
            "forward_arguments": [
                "images", "valid", "plane_ids", "fluid", "fat", "positions"
            ],
            "inference_mode": True,
            "probability_transform": "sigmoid(logits.float())",
            "all_valid_slices_processed": True,
            "normalization_performed_by_model": True,
        },
        "note": (
            "Software demonstration prediction for one gold_validation study. "
            "Not an independent validation estimate or a clinical diagnosis."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"csv": csv_path, "npz": npz_path, "report": report_path}
