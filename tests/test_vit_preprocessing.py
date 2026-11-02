import sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from rsna_knee.preprocessing.vit import (
    IMAGE_SIZE,
    MAX_TOTAL_SLICES,
    PLANE_ORDER,
    assemble_vit_study,
    normalize_resize_series,
)


def _fake_plane(value: int, plane_id: int):
    images = [np.full((256, 256), value + i, dtype=np.float32) for i in range(12)]
    normalized = normalize_resize_series(images)
    metadata = [
        {"plane_id": plane_id, "fluid": 1, "fat": 1, "position": i / 11}
        for i in range(12)
    ]
    return normalized, metadata, {
        "status": "ok", "n_files": 20, "n_decoded": 12,
        "series_uid": f"series-{plane_id}", "plane": PLANE_ORDER[plane_id],
    }


def test_normalize_resize_series_contract():
    images = [np.arange(256 * 256, dtype=np.float32).reshape(256, 256)]
    result = normalize_resize_series(images)
    assert len(result) == 1
    assert result[0].shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert result[0].dtype == np.uint8
    assert int(result[0].min()) >= 0
    assert int(result[0].max()) <= 255


def test_assemble_three_planes_to_fixed_36_slice_contract():
    outputs = {
        plane: _fake_plane(plane_id * 20, plane_id)
        for plane_id, plane in enumerate(PLANE_ORDER)
    }
    result = assemble_vit_study(
        outputs,
        study_uid="study-test",
        series_uids={plane: f"series-{i}" for i, plane in enumerate(PLANE_ORDER)},
    )
    assert result.images.shape == (MAX_TOTAL_SLICES, IMAGE_SIZE, IMAGE_SIZE)
    assert result.images.dtype == np.uint8
    assert result.valid.shape == (MAX_TOTAL_SLICES,)
    assert result.plane_ids.shape == (MAX_TOTAL_SLICES,)
    assert result.fluid.shape == (MAX_TOTAL_SLICES,)
    assert result.fat.shape == (MAX_TOTAL_SLICES,)
    assert result.positions.shape == (MAX_TOTAL_SLICES,)
    assert result.n_valid_slices == 36
    assert result.plane_ids.tolist() == [0] * 12 + [1] * 12 + [2] * 12


def test_assemble_partial_planes_keeps_fixed_shape_and_valid_mask():
    outputs = {
        "Sagittal": _fake_plane(10, 0),
        "Axial": _fake_plane(30, 2),
    }
    result = assemble_vit_study(
        outputs,
        study_uid="study-partial",
        series_uids={"Sagittal": "sag", "Coronal": "", "Axial": "ax"},
    )
    assert result.images.shape == (36, 224, 224)
    assert result.n_valid_slices == 24
    assert result.valid.tolist() == [1] * 24 + [0] * 12
    assert result.plane_ids.tolist() == [0] * 12 + [2] * 12 + [-1] * 12
    assert result.diagnostics[1]["status"] == "missing_selected_series"
