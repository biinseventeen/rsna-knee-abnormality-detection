"""Display-only DICOM decoding and thumbnail helpers for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _first_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, (str, bytes))
    ):
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
    """Convert decoded DICOM pixels into an 8-bit display image.

    This is only for the Streamlit viewer. It is not either model's inference
    preprocessing pipeline.
    """
    array = np.asarray(pixel_array)
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
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
        low, high = float(np.min(work)), float(np.max(work))
    if high <= low:
        result = np.zeros(work.shape, dtype=np.uint8)
    else:
        scaled = (np.clip(work, low, high) - low) / (high - low)
        result = np.asarray(np.round(scaled * 255.0), dtype=np.uint8)
    if str(photometric_interpretation).upper() == "MONOCHROME1":
        result = 255 - result
    return result


def read_dicom_preview(path: str | Path) -> np.ndarray:
    """Read and convert one DICOM image; pydicom is imported lazily."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Thiếu pydicom. Cài dependencies giao diện bằng `python -m pip install -r requirements-ui.txt`."
        ) from exc

    dicom_path = Path(path).expanduser().resolve()
    dataset = pydicom.dcmread(str(dicom_path), force=True)
    pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
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


def make_thumbnail(image: np.ndarray, max_side: int = 320) -> np.ndarray:
    """Resize an already-windowed image for tiled gallery display."""
    if max_side < 32:
        raise ValueError("max_side phải >= 32 pixel.")
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"Không hỗ trợ thumbnail shape {array.shape}.")
    height, width = array.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return array.astype(np.uint8, copy=False)

    scale = max_side / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Thiếu Pillow để tạo thumbnail. Cài requirements-ui.txt.") from exc
    image_obj = Image.fromarray(array)
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image_obj.resize(new_size, resampling)
    return np.asarray(resized, dtype=np.uint8)


def read_dicom_thumbnail(path: str | Path, max_side: int = 320) -> np.ndarray:
    """Decode one DICOM slice and return a small display thumbnail."""
    return make_thumbnail(read_dicom_preview(path), max_side=max_side)
