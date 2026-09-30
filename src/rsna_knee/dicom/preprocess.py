from __future__ import annotations

import numpy as np
from PIL import Image

from .io import load_dicom


def dicom_to_uint8(path) -> np.ndarray:
    ds = load_dicom(path)
    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    image = image * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image = image.max() - image

    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        lo, hi = float(image.min()), float(image.max())

    image = np.clip((image - lo) / max(hi - lo, 1e-6), 0, 1)
    return (image * 255).astype(np.uint8)


def make_rgb_stack(paths: list, image_size: int = 224) -> Image.Image:
    selected = list(paths[:3])
    if not selected:
        raise ValueError("Cannot build an image stack from zero DICOM slices.")

    while len(selected) < 3:
        selected.append(selected[-1])

    arrays = []
    for path in selected:
        arr = dicom_to_uint8(path)
        arr = np.asarray(Image.fromarray(arr).resize((image_size, image_size)))
        arrays.append(arr)

    rgb = np.stack(arrays, axis=-1)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")
