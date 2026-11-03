import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.ui.dicom_display import make_thumbnail, to_display_uint8  # noqa: E402


def test_grayscale_display_is_uint8_and_full_range():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    result = to_display_uint8(image)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_monochrome1_inverts_preview():
    image = np.array([[0, 10], [20, 30]], dtype=np.float32)
    normal = to_display_uint8(image, photometric_interpretation="MONOCHROME2")
    inverted = to_display_uint8(image, photometric_interpretation="MONOCHROME1")
    assert np.array_equal(inverted, 255 - normal)


def test_multi_frame_uses_middle_frame():
    frames = np.stack(
        [np.zeros((4, 5)), np.full((4, 5), 10), np.full((4, 5), 20)], axis=0
    )
    result = to_display_uint8(frames)
    assert result.shape == (4, 5)


def test_constant_image_does_not_divide_by_zero():
    result = to_display_uint8(np.ones((4, 4), dtype=np.float32))
    assert result.shape == (4, 4)
    assert np.isfinite(result).all()


def test_thumbnail_preserves_aspect_ratio_and_bounds_size():
    image = np.zeros((1000, 500), dtype=np.uint8)
    thumb = make_thumbnail(image, max_side=250)
    assert thumb.shape == (250, 125)
