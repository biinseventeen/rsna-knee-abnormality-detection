"""ViT-B/16 preprocessing ported from the final Kaggle ensemble notebook.

Source of truth:
    08_model_ensemble_kaggle_dataset_ready.ipynb
    Cell 7: one selected series per plane (selection is already exported in
            the local sample's series_selection.csv)
    Cell 9: DICOM ordering, pixel decoding, per-series percentile scaling,
            224x224 resizing, 12-slice sampling, and study packing.

This module deliberately mirrors the Kaggle path. It does not apply ImageNet
normalization; PretrainedViTStudyModel.forward() applies /255, RGB replication,
and ImageNet mean/std normalization internally.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..dicom.paths import long_path

IMAGE_SIZE = 224
MAX_TOTAL_SLICES = 36
SLICES_PER_SERIES = 12
MAX_SERIES_PER_STUDY = 3
MIN_DECODED_SLICES_PER_SERIES = 4
PLANE_ORDER = ("Sagittal", "Coronal", "Axial")
PLANE_TO_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


@dataclass(frozen=True)
class ViTPreprocessingResult:
    """One-study tensors matching the final ViT submission model input."""

    study_uid: str
    images: np.ndarray          # uint8 [36, 224, 224]
    valid: np.ndarray           # uint8 [36]
    plane_ids: np.ndarray       # int8 [36]
    fluid: np.ndarray           # int8 [36]
    fat: np.ndarray             # int8 [36]
    positions: np.ndarray       # float32 [36]
    series_uids: dict[str, str]
    diagnostics: list[dict[str, Any]]

    @property
    def n_valid_slices(self) -> int:
        return int(self.valid.sum())

    @property
    def n_series_used(self) -> int:
        return sum(row.get("status") == "ok" for row in self.diagnostics)

    def model_inputs(self) -> tuple[np.ndarray, ...]:
        return (
            self.images,
            self.valid,
            self.plane_ids,
            self.fluid,
            self.fat,
            self.positions,
        )


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _natural_key(name: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Thiếu pydicom. Cài bằng `python -m pip install 'pydicom>=3.0'`."
        ) from exc


def _apply_modality_lut(pixels: np.ndarray, dataset: Any) -> np.ndarray:
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:  # compatibility with older pydicom releases
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    try:
        return np.asarray(apply_modality_lut(pixels, dataset), dtype=np.float32)
    except Exception:
        slope = safe_float(getattr(dataset, "RescaleSlope", 1.0), 1.0)
        intercept = safe_float(getattr(dataset, "RescaleIntercept", 0.0), 0.0)
        return pixels.astype(np.float32) * slope + intercept


def read_sorted_dicom_headers(series_dir: str | Path) -> list[dict[str, Any]]:
    """Same slice ordering strategy and 70% threshold as Kaggle ViT Cell 9."""
    pydicom = _require_pydicom()
    series_dir = Path(series_dir)
    paths = sorted(
        p for p in series_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".dcm"
    )
    if not paths:
        return []

    rows: list[dict[str, Any]] = []
    first_normal = None
    for path in paths:
        try:
            ds = pydicom.dcmread(
                str(long_path(path)),
                stop_before_pixels=True,
                force=True,
                specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"],
            )
            iop = getattr(ds, "ImageOrientationPatient", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            instance = safe_float(getattr(ds, "InstanceNumber", np.nan))
            position = np.nan

            if iop is not None and ipp is not None and len(iop) >= 6 and len(ipp) >= 3:
                row_cos = np.asarray(iop[:3], dtype=np.float64)
                col_cos = np.asarray(iop[3:6], dtype=np.float64)
                normal = np.cross(row_cos, col_cos)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                    if first_normal is None:
                        first_normal = normal
                    position = float(
                        np.dot(np.asarray(ipp[:3], dtype=np.float64), first_normal)
                    )

            rows.append({
                "path": path,
                "filename": path.name,
                "position": position,
                "instance": instance,
            })
        except Exception as exc:
            rows.append({
                "path": path,
                "filename": path.name,
                "position": np.nan,
                "instance": np.nan,
                "header_error": f"{type(exc).__name__}: {exc}",
            })

    valid_pos = sum(np.isfinite(row.get("position", np.nan)) for row in rows)
    valid_inst = sum(np.isfinite(row.get("instance", np.nan)) for row in rows)

    if valid_pos >= max(2, int(0.7 * len(rows))):
        rows.sort(key=lambda row: (
            np.inf if not np.isfinite(row.get("position", np.nan)) else row["position"],
            row["filename"],
        ))
    elif valid_inst >= max(2, int(0.7 * len(rows))):
        rows.sort(key=lambda row: (
            np.inf if not np.isfinite(row.get("instance", np.nan)) else row["instance"],
            row["filename"],
        ))
    else:
        rows.sort(key=lambda row: _natural_key(row["filename"]))
    return rows


def decode_dicom_slice(path: str | Path) -> np.ndarray:
    """Decode one MRI slice using the exact final submission conversion."""
    pydicom = _require_pydicom()
    ds = pydicom.dcmread(str(long_path(path)), force=True)
    pixels = np.asarray(ds.pixel_array)

    if pixels.ndim == 3:
        if pixels.shape[-1] in (3, 4):
            pixels = pixels[..., :3].astype(np.float32)
            pixels = (
                0.299 * pixels[..., 0]
                + 0.587 * pixels[..., 1]
                + 0.114 * pixels[..., 2]
            )
        else:
            pixels = pixels[pixels.shape[0] // 2]

    pixels = np.squeeze(pixels)
    if pixels.ndim != 2:
        raise ValueError(f"Expected 2D MRI slice, got shape={pixels.shape}")

    pixels = _apply_modality_lut(pixels.astype(np.float32), ds)
    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        lo, hi = np.nanmin(pixels), np.nanmax(pixels)
        pixels = hi + lo - pixels

    pixels[~np.isfinite(pixels)] = 0
    return pixels.astype(np.float32, copy=False)


def normalize_resize_series(
    image_list: Sequence[np.ndarray], output_size: int = IMAGE_SIZE
) -> list[np.ndarray]:
    """Per-series percentile normalization and INTER_AREA resize (Kaggle Cell 9)."""
    if not image_list:
        return []
    import cv2

    all_values = np.concatenate([np.asarray(img).reshape(-1) for img in image_list])
    lo, hi = np.percentile(all_values, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(all_values)), float(np.max(all_values))
    if hi <= lo:
        hi = lo + 1.0

    output: list[np.ndarray] = []
    for img in image_list:
        x = np.clip(np.asarray(img, dtype=np.float32), lo, hi)
        x = (x - lo) / (hi - lo)
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).round().astype(np.uint8)
        x = cv2.resize(x, (output_size, output_size), interpolation=cv2.INTER_AREA)
        output.append(x)
    return output


def _metadata_flag(metadata: Mapping[str, Any], key: str) -> int:
    value = safe_float(metadata.get(key, -1), -1)
    return int(value) if value in (0, 1) else -1


def process_vit_series(series: Any, plane: str) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Process one already-selected plane series, matching Kaggle Cell 9."""
    if plane not in PLANE_TO_ID:
        raise ValueError(f"Unsupported plane {plane!r}; expected one of {PLANE_ORDER}.")

    headers = read_sorted_dicom_headers(series.directory)
    if not headers:
        return [], [], {
            "status": "no_dicom_files", "n_files": 0, "n_decoded": 0,
            "series_uid": str(series.series_uid), "plane": plane,
        }

    n_files = len(headers)
    target_n = min(SLICES_PER_SERIES, n_files)
    initial = np.unique(
        np.linspace(0, n_files - 1, target_n).round().astype(int)
    ).tolist()
    initial_set = set(initial)
    candidates = initial + [idx for idx in range(n_files) if idx not in initial_set]

    decoded: list[np.ndarray] = []
    decoded_positions: list[float] = []
    errors: list[str] = []
    for idx in candidates:
        try:
            image = decode_dicom_slice(headers[idx]["path"])
            decoded.append(image)
            decoded_positions.append(idx / max(n_files - 1, 1))
            if len(decoded) >= SLICES_PER_SERIES:
                break
        except Exception as exc:
            errors.append(
                f"{headers[idx]['filename']}: {type(exc).__name__}: {exc}"
            )

    if len(decoded) < MIN_DECODED_SLICES_PER_SERIES:
        return [], [], {
            "status": "too_few_decoded_slices",
            "n_files": n_files,
            "n_decoded": len(decoded),
            "error": " | ".join(errors[:2]),
            "series_uid": str(series.series_uid),
            "plane": plane,
        }

    order = np.argsort(decoded_positions)
    decoded = [decoded[int(i)] for i in order]
    decoded_positions = [decoded_positions[int(i)] for i in order]
    normalized = normalize_resize_series(decoded)

    plane_id = PLANE_TO_ID[plane]
    fluid = _metadata_flag(series.metadata, "Fluid_Sensitive")
    fat = _metadata_flag(series.metadata, "Fat_Suppression")
    metadata = [
        {
            "plane_id": plane_id,
            "fluid": fluid,
            "fat": fat,
            "position": float(position),
        }
        for position in decoded_positions
    ]

    return normalized, metadata, {
        "status": "ok",
        "n_files": n_files,
        "n_decoded": len(normalized),
        "error": " | ".join(errors[:3]),
        "series_uid": str(series.series_uid),
        "plane": plane,
        "sort_mode": _sort_mode_from_headers(headers),
    }


def _sort_mode_from_headers(headers: Sequence[Mapping[str, Any]]) -> str:
    if not headers:
        return "no_dicom_files"
    n = len(headers)
    valid_pos = sum(np.isfinite(row.get("position", np.nan)) for row in headers)
    valid_inst = sum(np.isfinite(row.get("instance", np.nan)) for row in headers)
    if valid_pos >= max(2, int(0.7 * n)):
        return "ImagePositionPatient"
    if valid_inst >= max(2, int(0.7 * n)):
        return "InstanceNumber"
    return "filename_fallback"


def assemble_vit_study(
    plane_outputs: Mapping[str, tuple[Sequence[np.ndarray], Sequence[Mapping[str, Any]], Mapping[str, Any]]],
    *,
    study_uid: str,
    series_uids: Mapping[str, str],
) -> ViTPreprocessingResult:
    """Pack per-plane outputs to the exact fixed-length 36-slice model input."""
    image_list: list[np.ndarray] = []
    metadata_list: list[Mapping[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    ordered_series_uids: dict[str, str] = {}

    for plane in PLANE_ORDER:
        series_uid = str(series_uids.get(plane, ""))
        ordered_series_uids[plane] = series_uid
        result = plane_outputs.get(plane)
        if result is None:
            diagnostics.append({
                "StudyInstanceUID": str(study_uid),
                "plane": plane,
                "series_uid": series_uid,
                "status": "missing_selected_series",
                "n_files": 0,
                "n_decoded": 0,
                "error": "",
            })
            continue

        images, metadata, info = result
        diagnostics.append({
            "StudyInstanceUID": str(study_uid),
            "plane": plane,
            "series_uid": series_uid,
            **dict(info),
        })
        if info.get("status") != "ok" or not images:
            continue
        if len(images) != len(metadata):
            raise ValueError(f"{plane}: image/metadata count mismatch.")
        if len(images) > SLICES_PER_SERIES:
            raise ValueError(f"{plane}: expected at most {SLICES_PER_SERIES} slices, got {len(images)}.")
        image_list.extend(images)
        metadata_list.extend(metadata)

    if not image_list:
        details = "; ".join(
            f"{row['plane']}={row.get('status')}" for row in diagnostics
        )
        raise RuntimeError(
            f"ViT preprocessing không tạo được slice hợp lệ cho study {study_uid}. {details}"
        )

    if len(image_list) > MAX_TOTAL_SLICES:
        image_list = image_list[:MAX_TOTAL_SLICES]
        metadata_list = metadata_list[:MAX_TOTAL_SLICES]

    images_out = np.zeros((MAX_TOTAL_SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    plane_ids = np.full(MAX_TOTAL_SLICES, -1, dtype=np.int8)
    fluid_flags = np.full(MAX_TOTAL_SLICES, -1, dtype=np.int8)
    fat_flags = np.full(MAX_TOTAL_SLICES, -1, dtype=np.int8)
    positions = np.zeros(MAX_TOTAL_SLICES, dtype=np.float32)
    valid = np.zeros(MAX_TOTAL_SLICES, dtype=np.uint8)

    n_valid = len(image_list)
    images_out[:n_valid] = np.stack(image_list, axis=0)
    valid[:n_valid] = 1
    for idx, meta in enumerate(metadata_list):
        plane_ids[idx] = int(meta["plane_id"])
        fluid_flags[idx] = int(meta["fluid"])
        fat_flags[idx] = int(meta["fat"])
        positions[idx] = float(meta["position"])

    return ViTPreprocessingResult(
        study_uid=str(study_uid),
        images=images_out,
        valid=valid,
        plane_ids=plane_ids,
        fluid=fluid_flags,
        fat=fat_flags,
        positions=positions,
        series_uids=ordered_series_uids,
        diagnostics=diagnostics,
    )


def preprocess_vit_study(study: Any) -> ViTPreprocessingResult:
    """Preprocess local ``DemoStudy`` using its Kaggle-exported ViT series map."""
    plane_outputs = {}
    series_uids: dict[str, str] = {}

    for plane in PLANE_ORDER:
        series = study.series_for_vit_plane(plane)
        if series is None:
            series_uids[plane] = ""
            plane_outputs[plane] = None
            continue
        series_uids[plane] = str(series.series_uid)
        plane_outputs[plane] = process_vit_series(series, plane)

    result = assemble_vit_study(
        plane_outputs,
        study_uid=str(study.study_uid),
        series_uids=series_uids,
    )
    if result.n_valid_slices == 0:
        raise RuntimeError("ViT preprocessing produced zero valid slices.")
    return result
