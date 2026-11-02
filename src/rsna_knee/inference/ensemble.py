"""Local 90/10 ensemble integration for the RSNA knee demo.

Consumes the already-computed outputs of Phase 2C (DINOv2 five-fold) and
Phase 2E (ViT-B/16 full-study). It does not load models or rerun inference.
The blend and per-model fallback policy match the final Kaggle ensemble:

- both valid: 0.9 * DINOv2 + 0.1 * ViT
- DINOv2 only: use DINOv2 probabilities unchanged
- ViT only: use ViT probabilities unchanged
- neither valid: neutral 0.5 fallback, explicitly marked as fallback

Inputs are validated for StudyInstanceUID, target order, shape, finite values,
probability range, model status and (for DINOv2) five-fold mean consistency.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

TARGETS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
N_TARGETS = len(TARGETS)
DEFAULT_DINO_WEIGHT = 0.90
DEFAULT_VIT_WEIGHT = 0.10


class EnsembleIntegrationError(RuntimeError):
    """Raised when the two model artifacts cannot be safely combined."""


@dataclass(frozen=True)
class ModelPrediction:
    model_name: str
    study_uid: str
    probabilities: np.ndarray
    valid: bool
    status: str
    source_npz: str
    source_report: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EnsemblePrediction:
    study_uid: str
    targets: tuple[str, ...]
    dinov2_probabilities: np.ndarray
    vit_probabilities: np.ndarray
    probabilities: np.ndarray
    dinov2_valid: bool
    vit_valid: bool
    configured_dinov2_weight: float
    configured_vit_weight: float
    effective_dinov2_weight: float
    effective_vit_weight: float
    blend_status: str
    dino_status: str
    vit_status: str


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Inference report not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EnsembleIntegrationError(f"Cannot read report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EnsembleIntegrationError(f"Report root must be an object: {path}")
    return value


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction NPZ not found: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}
    except Exception as exc:
        raise EnsembleIntegrationError(f"Cannot read prediction NPZ {path}: {exc}") from exc


def _scalar_text(value: np.ndarray, field: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise EnsembleIntegrationError(f"{field} must be a scalar, got shape {array.shape}.")
    return str(array.item())


def _validate_probability_vector(values: Any, model_name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float32)
    if probabilities.shape != (N_TARGETS,):
        raise EnsembleIntegrationError(
            f"{model_name} probabilities must have shape ({N_TARGETS},), "
            f"received {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise EnsembleIntegrationError(f"{model_name} probabilities contain NaN or Inf.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise EnsembleIntegrationError(f"{model_name} probabilities must be within [0, 1].")
    return probabilities


def load_dinov2_prediction(output_dir: str | Path) -> ModelPrediction:
    """Load and validate Phase 2C five-fold DINOv2 outputs."""
    root = Path(output_dir).expanduser().resolve()
    npz_path = root / "dinov2_five_fold_predictions.npz"
    report_path = root / "dinov2_five_fold_inference_report.json"
    arrays = _load_npz(npz_path)
    report = _read_json(report_path)

    required = {"study_uid", "targets", "status", "fold_probabilities", "ensemble_probabilities"}
    missing = sorted(required - set(arrays))
    if missing:
        raise EnsembleIntegrationError(f"DINOv2 NPZ is missing arrays: {missing}")

    study_uid = _scalar_text(arrays["study_uid"], "DINOv2 study_uid")
    npz_status = _scalar_text(arrays["status"], "DINOv2 status")
    targets = tuple(arrays["targets"].astype(str).tolist())
    if targets != TARGETS:
        raise EnsembleIntegrationError(
            "DINOv2 target order does not match the canonical competition order."
        )
    if str(report.get("study_instance_uid", "")) != study_uid:
        raise EnsembleIntegrationError("DINOv2 NPZ and report StudyInstanceUID differ.")
    if str(report.get("status", "")) != npz_status:
        raise EnsembleIntegrationError("DINOv2 NPZ status and report status differ.")

    probabilities = _validate_probability_vector(
        arrays["ensemble_probabilities"], "DINOv2"
    )
    fold_probabilities = np.asarray(arrays["fold_probabilities"], dtype=np.float32)
    if fold_probabilities.shape != (5, N_TARGETS):
        raise EnsembleIntegrationError(
            f"DINOv2 fold_probabilities must have shape (5, {N_TARGETS}), "
            f"received {fold_probabilities.shape}."
        )
    if not np.isfinite(fold_probabilities).all():
        raise EnsembleIntegrationError("DINOv2 fold probabilities contain NaN or Inf.")
    if np.any((fold_probabilities < 0.0) | (fold_probabilities > 1.0)):
        raise EnsembleIntegrationError("DINOv2 fold probabilities fall outside [0, 1].")

    # Recheck the exact Phase 2C aggregation: mean of fold sigmoid probabilities.
    fold_mean = fold_probabilities.mean(axis=0, dtype=np.float32)
    if not np.allclose(probabilities, fold_mean, rtol=1e-6, atol=1e-7):
        raise EnsembleIntegrationError(
            "DINOv2 ensemble probabilities do not equal the mean of its five fold probabilities."
        )

    if npz_status == "ok":
        fold_info = report.get("fold_info", [])
        if len(fold_info) != 5 or not all(
            isinstance(item, dict) and item.get("strict_load") is True
            for item in fold_info
        ):
            raise EnsembleIntegrationError(
                "DINOv2 report must confirm strict checkpoint loading for all five folds."
            )
        valid = True
    elif npz_status == "all_roles_unusable_fallback_0.5":
        valid = False
        if not np.allclose(probabilities, 0.5, rtol=0.0, atol=1e-7):
            raise EnsembleIntegrationError(
                "DINOv2 fallback status is set, but probabilities are not all 0.5."
            )
    else:
        raise EnsembleIntegrationError(f"Unknown DINOv2 inference status: {npz_status!r}.")

    return ModelPrediction(
        model_name="DINOv2",
        study_uid=study_uid,
        probabilities=probabilities,
        valid=valid,
        status=npz_status,
        source_npz=str(npz_path),
        source_report=str(report_path),
        metadata={
            "device": report.get("device"),
            "fold_count": int(fold_probabilities.shape[0]),
            "valid_windows": report.get("valid_windows"),
            "available_roles": report.get("available_roles"),
            "elapsed_seconds": report.get("elapsed_seconds"),
            "fold_info": report.get("fold_info", []),
        },
    )


def load_vit_prediction(output_dir: str | Path) -> ModelPrediction:
    """Load and validate Phase 2E full-study ViT outputs."""
    root = Path(output_dir).expanduser().resolve()
    npz_path = root / "vit_full_study_prediction.npz"
    report_path = root / "vit_full_study_inference_report.json"
    arrays = _load_npz(npz_path)
    report = _read_json(report_path)

    required = {"study_uid", "targets", "probabilities", "logits"}
    missing = sorted(required - set(arrays))
    if missing:
        raise EnsembleIntegrationError(f"ViT NPZ is missing arrays: {missing}")

    study_uid = _scalar_text(arrays["study_uid"], "ViT study_uid")
    targets = tuple(arrays["targets"].astype(str).tolist())
    if targets != TARGETS:
        raise EnsembleIntegrationError(
            "ViT target order does not match the canonical competition order."
        )
    if str(report.get("study_instance_uid", "")) != study_uid:
        raise EnsembleIntegrationError("ViT NPZ and report StudyInstanceUID differ.")

    probabilities = _validate_probability_vector(arrays["probabilities"], "ViT")
    logits = np.asarray(arrays["logits"], dtype=np.float32)
    if logits.shape != (N_TARGETS,) or not np.isfinite(logits).all():
        raise EnsembleIntegrationError("ViT logits must be 12 finite values.")

    input_info = report.get("input", {})
    valid_slices = int(input_info.get("valid_slices", 0))
    strict_load = report.get("strict_load") is True
    status = str(report.get("status", "unknown"))
    valid = status == "success" and strict_load and valid_slices > 0

    # A successful local ViT artifact must actually represent a full-study run.
    if status == "success":
        if not strict_load:
            raise EnsembleIntegrationError("ViT report says success but strict_load is not True.")
        if valid_slices <= 0:
            raise EnsembleIntegrationError("ViT report says success but has no valid slices.")
        output_info = report.get("output", {})
        if output_info.get("all_finite") is not True:
            raise EnsembleIntegrationError("ViT report does not confirm finite model outputs.")

    return ModelPrediction(
        model_name="ViT-B/16",
        study_uid=study_uid,
        probabilities=probabilities,
        valid=valid,
        status=status if valid else f"not_usable:{status}",
        source_npz=str(npz_path),
        source_report=str(report_path),
        metadata={
            "architecture": report.get("architecture"),
            "checkpoint_name": report.get("checkpoint_name"),
            "best_epoch": report.get("best_epoch"),
            "device": report.get("device"),
            "strict_load": strict_load,
            "valid_slices": valid_slices,
            "valid_slices_by_plane": input_info.get("valid_slices_by_plane", {}),
            "timing_seconds": report.get("timing_seconds", {}),
        },
    )


def blend_probabilities(
    *,
    study_uid: str,
    dinov2: ModelPrediction,
    vit: ModelPrediction,
    dinov2_weight: float = DEFAULT_DINO_WEIGHT,
) -> EnsemblePrediction:
    """Apply Kaggle's weighted blend and model-validity fallback policy."""
    if dinov2.study_uid != vit.study_uid or dinov2.study_uid != str(study_uid):
        raise EnsembleIntegrationError(
            "StudyInstanceUID mismatch between requested study and model predictions: "
            f"requested={study_uid}, DINOv2={dinov2.study_uid}, ViT={vit.study_uid}."
        )
    if not 0.0 <= float(dinov2_weight) <= 1.0:
        raise ValueError("dinov2_weight must be within [0, 1].")

    dino_probs = _validate_probability_vector(dinov2.probabilities, "DINOv2")
    vit_probs = _validate_probability_vector(vit.probabilities, "ViT")
    vit_weight = 1.0 - float(dinov2_weight)

    if dinov2.valid and vit.valid:
        effective_dino = float(dinov2_weight)
        effective_vit = float(vit_weight)
        blended = effective_dino * dino_probs + effective_vit * vit_probs
        status = "weighted_ensemble"
    elif dinov2.valid:
        effective_dino, effective_vit = 1.0, 0.0
        blended = dino_probs.copy()
        status = "dinov2_only"
    elif vit.valid:
        effective_dino, effective_vit = 0.0, 1.0
        blended = vit_probs.copy()
        status = "vit_only"
    else:
        effective_dino, effective_vit = 0.0, 0.0
        blended = np.full(N_TARGETS, 0.5, dtype=np.float32)
        status = "neutral_fallback_0.5"

    blended = np.asarray(blended, dtype=np.float32)
    if blended.shape != (N_TARGETS,) or not np.isfinite(blended).all():
        raise EnsembleIntegrationError("Final ensemble output is not a finite 12-value vector.")
    if np.any((blended < 0.0) | (blended > 1.0)):
        raise EnsembleIntegrationError("Final ensemble probabilities fall outside [0, 1].")

    return EnsemblePrediction(
        study_uid=str(study_uid),
        targets=TARGETS,
        dinov2_probabilities=dino_probs,
        vit_probabilities=vit_probs,
        probabilities=blended,
        dinov2_valid=bool(dinov2.valid),
        vit_valid=bool(vit.valid),
        configured_dinov2_weight=float(dinov2_weight),
        configured_vit_weight=float(vit_weight),
        effective_dinov2_weight=effective_dino,
        effective_vit_weight=effective_vit,
        blend_status=status,
        dino_status=dinov2.status,
        vit_status=vit.status,
    )


def save_ensemble_outputs(
    prediction: EnsemblePrediction,
    output_dir: str | Path,
    *,
    dino_source: ModelPrediction,
    vit_source: ModelPrediction,
) -> dict[str, Path]:
    """Save competition-shaped CSV, detailed CSV, NPZ and provenance JSON."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    wide_csv = output / "ensemble_prediction.csv"
    with wide_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", *TARGETS])
        writer.writeheader()
        row: dict[str, Any] = {"StudyInstanceUID": prediction.study_uid}
        row.update({target: float(prediction.probabilities[i]) for i, target in enumerate(TARGETS)})
        writer.writerow(row)

    long_csv = output / "ensemble_prediction_long.csv"
    with long_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "StudyInstanceUID", "target", "dinov2_probability", "vit_probability",
            "dinov2_valid", "vit_valid", "effective_dinov2_weight",
            "effective_vit_weight", "ensemble_probability", "blend_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, target in enumerate(TARGETS):
            writer.writerow({
                "StudyInstanceUID": prediction.study_uid,
                "target": target,
                "dinov2_probability": float(prediction.dinov2_probabilities[index]),
                "vit_probability": float(prediction.vit_probabilities[index]),
                "dinov2_valid": prediction.dinov2_valid,
                "vit_valid": prediction.vit_valid,
                "effective_dinov2_weight": prediction.effective_dinov2_weight,
                "effective_vit_weight": prediction.effective_vit_weight,
                "ensemble_probability": float(prediction.probabilities[index]),
                "blend_status": prediction.blend_status,
            })

    npz_path = output / "ensemble_prediction.npz"
    np.savez_compressed(
        npz_path,
        study_uid=np.asarray(prediction.study_uid),
        targets=np.asarray(prediction.targets),
        dinov2_probabilities=prediction.dinov2_probabilities.astype(np.float32),
        vit_probabilities=prediction.vit_probabilities.astype(np.float32),
        ensemble_probabilities=prediction.probabilities.astype(np.float32),
        dinov2_valid=np.asarray(prediction.dinov2_valid),
        vit_valid=np.asarray(prediction.vit_valid),
        configured_weights=np.asarray([
            prediction.configured_dinov2_weight,
            prediction.configured_vit_weight,
        ], dtype=np.float32),
        effective_weights=np.asarray([
            prediction.effective_dinov2_weight,
            prediction.effective_vit_weight,
        ], dtype=np.float32),
        blend_status=np.asarray(prediction.blend_status),
    )

    report_path = output / "ensemble_report.json"
    at_least_one_valid = prediction.dinov2_valid or prediction.vit_valid
    report = {
        "status": "success" if at_least_one_valid else "fallback",
        "blend_status": prediction.blend_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_instance_uid": prediction.study_uid,
        "targets": list(prediction.targets),
        "models": {
            "dinov2": {
                "valid": prediction.dinov2_valid,
                "status": prediction.dino_status,
                "source_npz": dino_source.source_npz,
                "source_report": dino_source.source_report,
                "metadata": dino_source.metadata,
            },
            "vit_b16": {
                "valid": prediction.vit_valid,
                "status": prediction.vit_status,
                "source_npz": vit_source.source_npz,
                "source_report": vit_source.source_report,
                "metadata": vit_source.metadata,
            },
        },
        "blend": {
            "configured_weights": {
                "dinov2": prediction.configured_dinov2_weight,
                "vit_b16": prediction.configured_vit_weight,
            },
            "effective_weights": {
                "dinov2": prediction.effective_dinov2_weight,
                "vit_b16": prediction.effective_vit_weight,
            },
            "policy": (
                "both valid: weighted blend; one valid: use that model alone; "
                "neither valid: neutral 0.5 fallback"
            ),
        },
        "output": {
            "shape": [N_TARGETS],
            "all_finite": bool(np.isfinite(prediction.probabilities).all()),
            "probability_min": float(prediction.probabilities.min()),
            "probability_max": float(prediction.probabilities.max()),
        },
        "files": {
            "wide_csv": wide_csv.name,
            "long_csv": long_csv.name,
            "npz": npz_path.name,
        },
        "note": (
            "One-study software demonstration output on the selected gold_validation sample. "
            "Not an independent validation estimate, a new competition score, or a clinical diagnosis."
        ),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"wide_csv": wide_csv, "long_csv": long_csv, "npz": npz_path, "report": report_path}


def run_local_ensemble(
    *,
    dinov2_output_dir: str | Path,
    vit_output_dir: str | Path,
    output_dir: str | Path,
    dinov2_weight: float = DEFAULT_DINO_WEIGHT,
) -> tuple[EnsemblePrediction, dict[str, Path]]:
    """Load Phase 2C/2E artifacts, verify identity/contracts and write the blend."""
    dino_source = load_dinov2_prediction(dinov2_output_dir)
    vit_source = load_vit_prediction(vit_output_dir)
    if dino_source.study_uid != vit_source.study_uid:
        raise EnsembleIntegrationError(
            "The two model artifacts belong to different studies: "
            f"DINOv2={dino_source.study_uid}, ViT={vit_source.study_uid}."
        )
    prediction = blend_probabilities(
        study_uid=dino_source.study_uid,
        dinov2=dino_source,
        vit=vit_source,
        dinov2_weight=dinov2_weight,
    )
    paths = save_ensemble_outputs(
        prediction,
        output_dir,
        dino_source=dino_source,
        vit_source=vit_source,
    )
    return prediction, paths
