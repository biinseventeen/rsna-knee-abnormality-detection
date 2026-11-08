"""Read-only, cross-validated readers for Phase 3 ensemble artifacts.

This module never loads model checkpoints and never executes inference. It
checks that all four Phase 3 outputs describe the same study and predictions
before the Streamlit UI is allowed to display them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..dicom.paths import long_path

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


class DashboardDataError(RuntimeError):
    """Raised when dashboard artifacts are missing or internally inconsistent."""


@dataclass(frozen=True)
class PredictionBundle:
    study_uid: str
    targets: tuple[str, ...]
    dinov2: np.ndarray
    vit: np.ndarray
    ensemble: np.ndarray
    long_table: pd.DataFrame
    report: dict[str, Any]
    output_dir: Path


def find_prediction_dir(output_root: str | Path, study_uid: str) -> Path:
    """Find the Phase 3 bundle for a study in a shared output directory.

    A single-study output directory remains supported. For multiple studies,
    bundles may be placed in ``<output_root>/<StudyInstanceUID>/`` (or any
    nested directory below the selected output root); the report UID selects
    the matching bundle.
    """
    root = long_path(output_root)
    if not root.is_dir():
        raise DashboardDataError(f"Ensemble output directory does not exist: {output_root}")

    direct_report = root / "ensemble_report.json"
    if direct_report.is_file():
        try:
            direct = json.loads(direct_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DashboardDataError(f"Cannot read ensemble report {direct_report}: {exc}") from exc
        if (
            isinstance(direct, dict)
            and str(direct.get("study_instance_uid", "")).strip()
            == str(study_uid).strip()
        ):
            return root

    candidates: set[Path] = set()
    for report_path in root.rglob("ensemble_report.json"):
        if report_path == direct_report:
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(report, dict)
            and str(report.get("study_instance_uid", "")).strip()
            == str(study_uid).strip()
        ):
            candidates.add(report_path.parent)

    if not candidates:
        raise DashboardDataError(
            f"No Phase 3 prediction bundle for study {study_uid!r} under {output_root}. "
            "A study can still be browsed without predictions."
        )
    if len(candidates) > 1:
        matches = "\n".join(str(path) for path in sorted(candidates, key=str))
        raise DashboardDataError(
            f"Multiple Phase 3 bundles match study {study_uid!r}:\n{matches}"
        )
    return next(iter(candidates))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DashboardDataError(f"Không tìm thấy report: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Không đọc được JSON report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DashboardDataError(f"JSON report phải có object ở root: {path}")
    return value


def _scalar_text(value: Any, name: str) -> str:
    arr = np.asarray(value)
    if arr.shape != ():
        raise DashboardDataError(f"NPZ field {name} phải là scalar, nhận shape {arr.shape}.")
    item = arr.item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    text = str(item).strip()
    if not text:
        raise DashboardDataError(f"NPZ field {name} đang rỗng.")
    return text


def _scalar_bool(value: Any, name: str) -> bool:
    arr = np.asarray(value)
    if arr.shape != () or arr.dtype.kind != "b":
        raise DashboardDataError(f"NPZ field {name} phải là boolean scalar.")
    return bool(arr.item())


def _vector(values: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DashboardDataError(f"{name} không phải dữ liệu số hợp lệ.") from exc
    if arr.shape != (N_TARGETS,):
        raise DashboardDataError(
            f"{name} cần đúng {N_TARGETS} giá trị, nhận được shape {arr.shape}."
        )
    if not np.isfinite(arr).all():
        raise DashboardDataError(f"{name} chứa NaN hoặc Inf.")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise DashboardDataError(f"{name} có xác suất nằm ngoài [0, 1].")
    return arr


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise DashboardDataError(f"Không tìm thấy NPZ Phase 3: {path}")
    required = {
        "study_uid",
        "targets",
        "dinov2_probabilities",
        "vit_probabilities",
        "ensemble_probabilities",
        "dinov2_valid",
        "vit_valid",
        "configured_weights",
        "effective_weights",
        "blend_status",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise DashboardDataError(f"NPZ Phase 3 thiếu arrays: {missing}")
            return {name: np.asarray(archive[name]).copy() for name in required}
    except DashboardDataError:
        raise
    except Exception as exc:
        raise DashboardDataError(f"Không đọc được NPZ Phase 3 {path}: {exc}") from exc


def _csv_bool_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    parsed: list[bool] = []
    for raw in frame[column].tolist():
        if isinstance(raw, (bool, np.bool_)):
            parsed.append(bool(raw))
        elif str(raw).strip().casefold() in {"true", "1"}:
            parsed.append(True)
        elif str(raw).strip().casefold() in {"false", "0"}:
            parsed.append(False)
        else:
            raise DashboardDataError(
                f"Cột {column} có giá trị boolean không hợp lệ: {raw!r}."
            )
    return np.asarray(parsed, dtype=bool)


def load_prediction_bundle(output_dir: str | Path) -> PredictionBundle:
    """Read and cross-check all four Phase 3 artifacts for one demo study."""
    root = long_path(output_dir)
    wide_path = root / "ensemble_prediction.csv"
    long_csv_path = root / "ensemble_prediction_long.csv"
    npz_path = root / "ensemble_prediction.npz"
    report_path = root / "ensemble_report.json"

    for path in (wide_path, long_csv_path, npz_path, report_path):
        if not path.is_file():
            raise DashboardDataError(f"Thiếu output Phase 3: {path}")

    try:
        wide = pd.read_csv(wide_path)
        long_table = pd.read_csv(long_csv_path)
    except Exception as exc:
        raise DashboardDataError(f"Không đọc được CSV output Phase 3: {exc}") from exc
    arrays = _read_npz(npz_path)
    report = _read_json(report_path)

    # Wide competition-shaped row and its strict target order.
    if len(wide) != 1:
        raise DashboardDataError(
            f"ensemble_prediction.csv cần đúng 1 study row, hiện có {len(wide)}."
        )
    if not wide.columns.tolist() == ["StudyInstanceUID", *TARGETS]:
        raise DashboardDataError(
            "Cột hoặc thứ tự target trong wide CSV không khớp canonical 12-target order."
        )

    # Detailed per-target CSV.
    expected_long_columns = {
        "StudyInstanceUID",
        "target",
        "dinov2_probability",
        "vit_probability",
        "ensemble_probability",
        "dinov2_valid",
        "vit_valid",
        "effective_dinov2_weight",
        "effective_vit_weight",
        "blend_status",
    }
    missing_long = sorted(expected_long_columns - set(long_table.columns))
    if missing_long:
        raise DashboardDataError(f"Long CSV thiếu cột: {missing_long}")
    if len(long_table) != N_TARGETS:
        raise DashboardDataError(
            f"Long CSV cần đúng {N_TARGETS} target rows, hiện có {len(long_table)}."
        )
    targets = tuple(long_table["target"].astype(str).tolist())
    if targets != TARGETS:
        raise DashboardDataError(
            "Target order trong long CSV không khớp canonical 12-target order."
        )

    # One StudyInstanceUID must be shared by CSV, NPZ, and JSON.
    study_uid = str(wide.loc[0, "StudyInstanceUID"]).strip()
    report_uid = str(report.get("study_instance_uid", "")).strip()
    npz_uid = _scalar_text(arrays["study_uid"], "study_uid")
    if not study_uid or study_uid != report_uid or study_uid != npz_uid:
        raise DashboardDataError(
            "StudyInstanceUID không khớp giữa wide CSV, NPZ và JSON report: "
            f"wide={study_uid!r}, npz={npz_uid!r}, report={report_uid!r}."
        )
    if set(long_table["StudyInstanceUID"].astype(str)) != {study_uid}:
        raise DashboardDataError("Long CSV có StudyInstanceUID không khớp với wide CSV.")

    report_targets = tuple(report.get("targets", ()))
    npz_targets = tuple(np.asarray(arrays["targets"]).astype(str).tolist())
    if report_targets != TARGETS or npz_targets != TARGETS:
        raise DashboardDataError(
            "Canonical target order không khớp giữa JSON/NPZ và 12 target chuẩn."
        )

    # Probability vectors from all three numerical representations.
    dino = _vector(long_table["dinov2_probability"].to_numpy(), "DINOv2 CSV")
    vit = _vector(long_table["vit_probability"].to_numpy(), "ViT CSV")
    ensemble = _vector(long_table["ensemble_probability"].to_numpy(), "Ensemble CSV")
    wide_ensemble = _vector(wide.loc[0, list(TARGETS)].to_numpy(), "Wide CSV")
    npz_dino = _vector(arrays["dinov2_probabilities"], "DINOv2 NPZ")
    npz_vit = _vector(arrays["vit_probabilities"], "ViT NPZ")
    npz_ensemble = _vector(arrays["ensemble_probabilities"], "Ensemble NPZ")

    for label, left, right in (
        ("DINOv2 CSV/NPZ", dino, npz_dino),
        ("ViT CSV/NPZ", vit, npz_vit),
        ("Ensemble CSV/wide CSV", ensemble, wide_ensemble),
        ("Ensemble CSV/NPZ", ensemble, npz_ensemble),
    ):
        if not np.allclose(left, right, rtol=1e-6, atol=1e-7):
            raise DashboardDataError(f"Prediction values không khớp: {label}.")

    # Model validity must agree between JSON, long CSV and NPZ.
    models = report.get("models", {})
    blend = report.get("blend", {})
    if not isinstance(models, dict) or not isinstance(blend, dict):
        raise DashboardDataError("Report thiếu object models hoặc blend.")
    dino_report = models.get("dinov2", {})
    vit_report = models.get("vit_b16", {})
    if not isinstance(dino_report, dict) or not isinstance(vit_report, dict):
        raise DashboardDataError("Report thiếu metadata của DINOv2 hoặc ViT-B/16.")
    report_dino_valid = dino_report.get("valid")
    report_vit_valid = vit_report.get("valid")
    if not isinstance(report_dino_valid, bool) or not isinstance(report_vit_valid, bool):
        raise DashboardDataError("Report không có trạng thái valid boolean của cả hai model.")

    dino_valid_rows = _csv_bool_column(long_table, "dinov2_valid")
    vit_valid_rows = _csv_bool_column(long_table, "vit_valid")
    npz_dino_valid = _scalar_bool(arrays["dinov2_valid"], "dinov2_valid")
    npz_vit_valid = _scalar_bool(arrays["vit_valid"], "vit_valid")
    if not np.all(dino_valid_rows == report_dino_valid) or npz_dino_valid != report_dino_valid:
        raise DashboardDataError("Trạng thái valid của DINOv2 không khớp giữa CSV/NPZ/report.")
    if not np.all(vit_valid_rows == report_vit_valid) or npz_vit_valid != report_vit_valid:
        raise DashboardDataError("Trạng thái valid của ViT không khớp giữa CSV/NPZ/report.")

    # Blend state and effective weights must be identical across artifacts.
    report_status = str(report.get("blend_status", ""))
    long_statuses = set(long_table["blend_status"].astype(str))
    npz_status = _scalar_text(arrays["blend_status"], "blend_status")
    if long_statuses != {report_status} or npz_status != report_status:
        raise DashboardDataError("blend_status không khớp giữa CSV, NPZ và JSON report.")

    weights = blend.get("effective_weights", {})
    configured = blend.get("configured_weights", {})
    try:
        dino_weight = float(weights["dinov2"])
        vit_weight = float(weights["vit_b16"])
        configured_dino_weight = float(configured["dinov2"])
        configured_vit_weight = float(configured["vit_b16"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DashboardDataError("Report thiếu configured/effective ensemble weights.") from exc
    if not np.isfinite([dino_weight, vit_weight, configured_dino_weight, configured_vit_weight]).all():
        raise DashboardDataError("Ensemble weights trong report chứa NaN hoặc Inf.")

    try:
        csv_dino_weights = long_table["effective_dinov2_weight"].to_numpy(dtype=np.float64)
        csv_vit_weights = long_table["effective_vit_weight"].to_numpy(dtype=np.float64)
        npz_effective = np.asarray(arrays["effective_weights"], dtype=np.float64)
        npz_configured = np.asarray(arrays["configured_weights"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DashboardDataError("Effective/configured weights không phải dữ liệu số.") from exc
    if npz_effective.shape != (2,) or npz_configured.shape != (2,):
        raise DashboardDataError("NPZ weights phải có shape (2,): [DINOv2, ViT].")
    if not np.isfinite(csv_dino_weights).all() or not np.isfinite(csv_vit_weights).all():
        raise DashboardDataError("Effective weights trong long CSV chứa NaN hoặc Inf.")
    expected_effective = np.asarray([dino_weight, vit_weight], dtype=np.float64)
    expected_configured = np.asarray(
        [configured_dino_weight, configured_vit_weight], dtype=np.float64
    )
    if not np.allclose(csv_dino_weights, dino_weight, rtol=0.0, atol=1e-7) or not np.allclose(
        csv_vit_weights, vit_weight, rtol=0.0, atol=1e-7
    ):
        raise DashboardDataError("Effective weights trong CSV và JSON report không khớp.")
    if not np.allclose(npz_effective, expected_effective, rtol=0.0, atol=1e-7):
        raise DashboardDataError("Effective weights trong NPZ và JSON report không khớp.")
    if not np.allclose(npz_configured, expected_configured, rtol=0.0, atol=1e-7):
        raise DashboardDataError("Configured weights trong NPZ và JSON report không khớp.")

    # Re-apply Phase 3 policy from the recorded validity flags.
    if report_status == "weighted_ensemble":
        if not (report_dino_valid and report_vit_valid):
            raise DashboardDataError("weighted_ensemble nhưng có model không valid.")
        if not np.allclose(expected_effective, [0.9, 0.1], atol=1e-7):
            raise DashboardDataError("Trọng số weighted ensemble không khớp 90/10.")
    elif report_status == "dinov2_only":
        if not report_dino_valid or report_vit_valid or not np.allclose(
            expected_effective, [1.0, 0.0], atol=1e-7
        ):
            raise DashboardDataError("DINOv2-only fallback không nhất quán.")
    elif report_status == "vit_only":
        if report_dino_valid or not report_vit_valid or not np.allclose(
            expected_effective, [0.0, 1.0], atol=1e-7
        ):
            raise DashboardDataError("ViT-only fallback không nhất quán.")
    elif report_status == "neutral_fallback_0.5":
        if report_dino_valid or report_vit_valid or not np.allclose(
            expected_effective, [0.0, 0.0], atol=1e-7
        ) or not np.allclose(ensemble, 0.5, atol=1e-7):
            raise DashboardDataError("Neutral 0.5 fallback không nhất quán.")
    else:
        raise DashboardDataError(f"blend_status không được hỗ trợ: {report_status!r}.")

    expected_report_status = "fallback" if report_status == "neutral_fallback_0.5" else "success"
    if str(report.get("status", "")) != expected_report_status:
        raise DashboardDataError("Top-level report status không khớp blend_status.")

    if report_status == "neutral_fallback_0.5":
        expected_ensemble = np.full(N_TARGETS, 0.5, dtype=np.float64)
    else:
        expected_ensemble = dino_weight * dino + vit_weight * vit
    if not np.allclose(ensemble, expected_ensemble, rtol=1e-6, atol=1e-7):
        raise DashboardDataError("Ensemble probabilities không khớp công thức và effective weights.")

    return PredictionBundle(
        study_uid=study_uid,
        targets=TARGETS,
        dinov2=dino,
        vit=vit,
        ensemble=ensemble,
        long_table=long_table,
        report=report,
        output_dir=root,
    )
