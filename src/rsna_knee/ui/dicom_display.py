"""Small DICOM display helper for the dashboard (not model preprocessing)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _first_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        try:
            value = list(value)[0]
        except (TypeError, IndexError, KeyError):
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def to_display_uint8(
    pixel_array: np.ndarray,
    *,
    photometric_interpretation: str = "MONOCHROME2",
    window_center: float | None = None,
    window_width: float | None = None,
) -> np.ndarray:
    """Convert a decoded DICOM array to an 8-bit preview image.

    Uses DICOM window center/width when both are valid; otherwise it uses the
    1st and 99th percentiles. This is for visual display only and is not a
    replacement for either model's Kaggle-matched preprocessing pipeline.
    """
    array = np.asarray(pixel_array)
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        # Multi-frame grayscale DICOM: show the middle frame.
        array = array[array.shape[0] // 2]
    if array.ndim not in (2, 3):
        raise ValueError(f"Không hỗ trợ pixel array shape {array.shape} để preview.")

    work = np.asarray(array, dtype=np.float32)
    work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)

    if window_center is not None and window_width is not None and window_width > 1:
        low = float(window_center) - float(window_width) / 2.0
        high = float(window_center) + float(window_width) / 2.0
    else:
        low = float(np.percentile(work, 1.0))
        high = float(np.percentile(work, 99.0))

    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(work))
        high = float(np.max(work))
    if high <= low:
        result = np.zeros(work.shape, dtype=np.uint8)
    else:
        scaled = (np.clip(work, low, high) - low) / (high - low)
        result = np.asarray(np.round(scaled * 255.0), dtype=np.uint8)

    if str(photometric_interpretation).upper() == "MONOCHROME1":
        result = 255 - result
    return result


def read_dicom_preview(path: str | Path) -> np.ndarray:
    """Read and convert one DICOM file for display; imports pydicom lazily."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - requires local optional dependency
        raise RuntimeError(
            "Thiếu pydicom. Cài dependencies giao diện bằng `python -m pip install -r requirements-ui.txt`."
        ) from exc

    dicom_path = Path(path).expanduser().resolve()
    dataset = pydicom.dcmread(str(dicom_path), force=True)
    pixels = dataset.pixel_array
    pixels = np.asarray(pixels, dtype=np.float32)
    slope = _first_number(getattr(dataset, "RescaleSlope", 1.0))
    intercept = _first_number(getattr(dataset, "RescaleIntercept", 0.0))
    pixels = pixels * (1.0 if slope is None else slope) + (0.0 if intercept is None else intercept)

    center = _first_number(getattr(dataset, "WindowCenter", None))
    width = _first_number(getattr(dataset, "WindowWidth", None))
    photo = str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"))
    return to_display_uint8(
        pixels,
        photometric_interpretation=photo,
        window_center=center,
        window_width=width,
    )
