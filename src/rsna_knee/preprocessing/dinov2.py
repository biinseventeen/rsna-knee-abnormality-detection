"""DINOv2 five-role 2.5D preprocessing ported from the final Kaggle ensemble.

Source of truth:
    Notebook 08_model_ensemble_kaggle_dataset_ready.ipynb
    Sections 12-14: constants/role selection, DICOM preprocessing, 2.5D windows.

This module intentionally returns uint8 tensors. The model's forward() applies
[0, 1] scaling and ImageNet normalization, so preprocessing must not normalize
these images a second time.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

IMAGE_SIZE = 224
PATCH_SIZE = 14
N_ROLE_SLICES = 12
N_ROLE_WINDOWS = N_ROLE_SLICES - 2
MIN_DECODED_SLICES = 4
FIVE_ROLE_ORDER = (
    "SAG_FLUID",
    "COR_FLUID",
    "AX_FLUID",
    "SAG_NONFLUID",
    "COR_NONFLUID",
)

DICOM_HEADER_TAGS = [
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "InstanceNumber",
    "SOPInstanceUID",
    "PixelSpacing",
    "Rows",
    "Columns",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "Laterality",
    "ImageLaterality",
    "FrameOfReferenceUID",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PhotometricInterpretation",
    "RescaleSlope",
    "RescaleIntercept",
    "StudyInstanceUID",
    "SeriesInstanceUID",
]


@dataclass(frozen=True)
class DINOv2PreprocessingResult:
    """Five-role input tensors and diagnostics for one study.

    images:
        uint8 array [5, 10, 3, 224, 224]. No batch dimension.
    window_valid / window_positions:
        arrays [5, 10].
    """

    study_uid: str
    images: np.ndarray
    window_valid: np.ndarray
    window_positions: np.ndarray
    role_series_uids: dict[str, str]
    diagnostics: list[dict[str, Any]]

    @property
    def n_valid_windows(self) -> int:
        return int(self.window_valid.sum())

    @property
    def n_available_roles(self) -> int:
        return int(self.window_valid.any(axis=1).sum())

    def model_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return arrays in the exact per-study layout expected by the model."""
        return self.images, self.window_valid, self.window_positions


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def natural_key(name: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def read_sorted_dicom_headers(series_dir: Path) -> tuple[list[dict[str, Any]], str]:
    """Read headers and sort slices exactly as the final Kaggle DINOv2 path."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Thiếu pydicom. Cài bằng `python -m pip install 'pydicom>=3.0'`."
        ) from exc

    series_dir = Path(series_dir)
    files = sorted(
        path for path in series_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".dcm"
    )
    if not files:
        files = sorted(
            path for path in series_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )

    rows: list[dict[str, Any]] = []
    first_normal: np.ndarray | None = None
    for path in files:
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=DICOM_HEADER_TAGS,
            )
            iop = getattr(ds, "ImageOrientationPatient", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            position = math.nan
            instance = safe_float(getattr(ds, "InstanceNumber", math.nan))
            if iop is not None and ipp is not None and len(iop) >= 6 and len(ipp) >= 3:
                normal = np.cross(
                    np.asarray(iop[:3], dtype=np.float64),
                    np.asarray(iop[3:6], dtype=np.float64),
                )
                norm = float(np.linalg.norm(normal))
                if norm > 0:
                    normal = normal / norm
                    if first_normal is None:
                        first_normal = normal
                    position = float(
                        np.dot(np.asarray(ipp[:3], dtype=np.float64), first_normal)
                    )
            rows.append(
                {
                    "path": path,
                    "filename": path.name,
                    "position_mm": position,
                    "instance": instance,
                    "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "path": path,
                    "filename": path.name,
                    "position_mm": math.nan,
                    "instance": math.nan,
                    "sop_uid": "",
                    "header_error": f"{type(exc).__name__}: {exc}",
                }
            )

    if not rows:
        return [], "no_dicom_files"

    n = len(rows)
    n_position = sum(math.isfinite(row["position_mm"]) for row in rows)
    n_instance = sum(math.isfinite(row["instance"]) for row in rows)
    if n_position >= max(2, int(0.7 * n)):
        rows.sort(
            key=lambda row: (
                math.inf if not math.isfinite(row["position_mm"]) else row["position_mm"],
                row["filename"],
            )
        )
        sort_mode = "ImagePositionPatient_projected"
    elif n_instance >= max(2, int(0.7 * n)):
        rows.sort(
            key=lambda row: (
                math.inf if not math.isfinite(row["instance"]) else row["instance"],
                row["filename"],
            )
        )
        sort_mode = "InstanceNumber"
    else:
        rows.sort(key=lambda row: natural_key(row["filename"]))
        sort_mode = "filename_fallback"

    for index, row in enumerate(rows):
        row["sort_index"] = index
    return rows, sort_mode


def decode_dicom_slice(path: Path) -> tuple[np.ndarray, Any]:
    """Decode one DICOM slice using the final Kaggle preprocessing operations."""
    try:
        import pydicom
        from pydicom.pixels import apply_modality_lut
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Thiếu pydicom. Cài bằng `python -m pip install 'pydicom>=3.0'`."
        ) from exc

    ds = pydicom.dcmread(str(path), force=True)
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

    pixels = pixels.astype(np.float32)
    try:
        pixels = np.asarray(apply_modality_lut(pixels, ds), dtype=np.float32)
    except Exception:
        slope = safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        pixels = pixels * slope + intercept

    if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper() == "MONOCHROME1":
        lo, hi = np.nanmin(pixels), np.nanmax(pixels)
        pixels = hi + lo - pixels
    pixels[~np.isfinite(pixels)] = 0
    return pixels, ds


def process_role_series(
    study_uid: str,
    series_uid: str,
    series_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Decode one series to 12 uint8 224x224 slices, matching ensemble Cell 25."""
    zero_images = np.zeros((N_ROLE_SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    zero_valid = np.zeros(N_ROLE_SLICES, dtype=np.uint8)
    zero_positions = np.zeros(N_ROLE_SLICES, dtype=np.float32)

    series_dir = Path(series_dir)
    if not series_dir.is_dir():
        return None, None, None, {
            "status": "missing_series_directory", "n_files": 0, "n_decoded": 0
        }

    headers, sort_mode = read_sorted_dicom_headers(series_dir)
    if not headers:
        return None, None, None, {
            "status": sort_mode, "n_files": 0, "n_decoded": 0
        }

    n_files = len(headers)
    initial_indices = np.unique(
        np.linspace(0, n_files - 1, min(N_ROLE_SLICES, n_files)).round().astype(int)
    ).tolist()
    initial_set = set(initial_indices)
    candidates = initial_indices + [i for i in range(n_files) if i not in initial_set]

    decoded_images: list[np.ndarray] = []
    decoded_positions: list[float] = []
    errors: list[str] = []

    for header_index in candidates:
        header = headers[header_index]
        try:
            image, ds = decode_dicom_slice(header["path"])
            actual_study = str(getattr(ds, "StudyInstanceUID", ""))
            actual_series = str(getattr(ds, "SeriesInstanceUID", ""))
            if actual_study and actual_study != str(study_uid):
                raise ValueError("StudyInstanceUID mismatch")
            if actual_series and actual_series != str(series_uid):
                raise ValueError("SeriesInstanceUID mismatch")

            decoded_images.append(image)
            decoded_positions.append(header["sort_index"] / max(n_files - 1, 1))
            if len(decoded_images) >= N_ROLE_SLICES:
                break
        except Exception as exc:
            errors.append(f"{header['filename']}:{type(exc).__name__}:{exc}")

    if len(decoded_images) < MIN_DECODED_SLICES:
        return None, None, None, {
            "status": "too_few_decoded_slices",
            "n_files": n_files,
            "n_decoded": len(decoded_images),
            "sort_mode": sort_mode,
            "error": " | ".join(errors[:2]),
        }

    order = np.argsort(decoded_positions, kind="stable")
    decoded_images = [decoded_images[index] for index in order]
    decoded_positions = [decoded_positions[index] for index in order]

    values = np.concatenate([image.reshape(-1) for image in decoded_images])
    low, high = np.percentile(values, [1.0, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        high = low + 1.0

    out_images = zero_images.copy()
    out_valid = zero_valid.copy()
    out_positions = zero_positions.copy()
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Thiếu OpenCV. Cài bằng `python -m pip install opencv-python-headless`."
        ) from exc

    for slot, (image, position) in enumerate(zip(decoded_images, decoded_positions)):
        scaled = np.clip((np.clip(image, low, high) - low) / (high - low), 0, 1)
        scaled = (scaled * 255.0).round().astype(np.uint8)
        out_images[slot] = cv2.resize(
            scaled, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA
        )
        out_valid[slot] = 1
        out_positions[slot] = float(position)

    return out_images, out_valid, out_positions, {
        "status": "ok",
        "n_files": n_files,
        "n_decoded": len(decoded_images),
        "sort_mode": sort_mode,
        "error": " | ".join(errors[:3]),
        "percentile_low": float(low),
        "percentile_high": float(high),
    }


def assemble_five_role_windows(
    role_arrays: Mapping[
        str,
        tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]],
    ],
    *,
    study_uid: str,
    role_series_uids: Mapping[str, str],
) -> DINOv2PreprocessingResult:
    """Build overlapping 3-slice windows exactly as ensemble Cell 27."""
    images = np.zeros(
        (len(FIVE_ROLE_ORDER), N_ROLE_WINDOWS, 3, IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8,
    )
    window_valid = np.zeros((len(FIVE_ROLE_ORDER), N_ROLE_WINDOWS), dtype=np.uint8)
    window_positions = np.zeros((len(FIVE_ROLE_ORDER), N_ROLE_WINDOWS), dtype=np.float32)
    diagnostics: list[dict[str, Any]] = []

    for role_index, role in enumerate(FIVE_ROLE_ORDER):
        role_result = role_arrays.get(role)
        if role_result is None:
            diagnostics.append(
                {
                    "StudyInstanceUID": str(study_uid),
                    "role": role,
                    "SeriesInstanceUID": str(role_series_uids.get(role, "")),
                    "status": "no_candidate_series",
                    "n_files": 0,
                    "n_decoded": 0,
                }
            )
            continue

        slices, slice_valid, slice_positions, info = role_result
        diagnostics.append(
            {
                "StudyInstanceUID": str(study_uid),
                "role": role,
                "SeriesInstanceUID": str(role_series_uids.get(role, "")),
                **info,
            }
        )
        if slices is None or slice_valid is None or slice_positions is None:
            continue

        expected_slices = (N_ROLE_SLICES, IMAGE_SIZE, IMAGE_SIZE)
        if slices.shape != expected_slices:
            raise ValueError(f"{role}: unexpected slices shape {slices.shape}; expected {expected_slices}")
        if slice_valid.shape != (N_ROLE_SLICES,) or slice_positions.shape != (N_ROLE_SLICES,):
            raise ValueError(f"{role}: slice validity/position arrays must have shape ({N_ROLE_SLICES},).")

        for window_index in range(N_ROLE_WINDOWS):
            local_valid = slice_valid[window_index : window_index + 3]
            if int(local_valid.sum()) >= 2:
                images[role_index, window_index] = slices[window_index : window_index + 3]
                window_valid[role_index, window_index] = 1
                valid_mask = local_valid.astype(bool)
                window_positions[role_index, window_index] = float(
                    slice_positions[window_index : window_index + 3][valid_mask].mean()
                )

    return DINOv2PreprocessingResult(
        study_uid=str(study_uid),
        images=images,
        window_valid=window_valid,
        window_positions=window_positions,
        role_series_uids={role: str(role_series_uids.get(role, "")) for role in FIVE_ROLE_ORDER},
        diagnostics=diagnostics,
    )


def preprocess_dinov2_study(study: Any) -> DINOv2PreprocessingResult:
    """Preprocess a ``DemoStudy`` returned by ``load_demo_sample``.

    Role-to-series mapping is read from ``series_selection.csv`` via
    ``study.series_for_dino_role``. It is not re-inferred from DICOM filenames.
    """
    role_arrays: dict[str, tuple[Any, Any, Any, dict[str, Any]]] = {}
    role_series_uids: dict[str, str] = {}

    for role in FIVE_ROLE_ORDER:
        series = study.series_for_dino_role(role)
        if series is None:
            role_series_uids[role] = ""
            role_arrays[role] = (
                None,
                None,
                None,
                {"status": "no_candidate_series", "n_files": 0, "n_decoded": 0},
            )
            continue

        role_series_uids[role] = str(series.series_uid)
        role_arrays[role] = process_role_series(
            study.study_uid,
            series.series_uid,
            series.directory,
        )

    result = assemble_five_role_windows(
        role_arrays,
        study_uid=study.study_uid,
        role_series_uids=role_series_uids,
    )
    if result.n_valid_windows == 0:
        raise RuntimeError(
            "DINOv2 preprocessing không tạo được valid window nào. "
            "Kiểm tra series_selection.csv và pixel decoding diagnostics."
        )
    return result
