import numpy as np

from rsna_knee.preprocessing.dinov2 import (
    FIVE_ROLE_ORDER,
    IMAGE_SIZE,
    N_ROLE_SLICES,
    N_ROLE_WINDOWS,
    assemble_five_role_windows,
)


def _role_arrays(offset: int = 0):
    slices = np.empty((N_ROLE_SLICES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for index in range(N_ROLE_SLICES):
        slices[index].fill((index + offset) % 255)
    valid = np.ones(N_ROLE_SLICES, dtype=np.uint8)
    positions = np.linspace(0.0, 1.0, N_ROLE_SLICES, dtype=np.float32)
    return slices, valid, positions, {"status": "ok", "n_files": 36, "n_decoded": 12}


def test_five_role_window_shapes_and_overlap():
    role_arrays = {role: _role_arrays(i * 10) for i, role in enumerate(FIVE_ROLE_ORDER)}
    role_uids = {role: f"series-{i}" for i, role in enumerate(FIVE_ROLE_ORDER)}

    result = assemble_five_role_windows(
        role_arrays,
        study_uid="study-1",
        role_series_uids=role_uids,
    )

    assert result.images.shape == (5, N_ROLE_WINDOWS, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert result.window_valid.shape == (5, N_ROLE_WINDOWS)
    assert result.window_positions.shape == (5, N_ROLE_WINDOWS)
    assert result.images.dtype == np.uint8
    assert result.window_valid.dtype == np.uint8
    assert result.window_positions.dtype == np.float32
    assert result.n_valid_windows == 5 * N_ROLE_WINDOWS
    assert result.n_available_roles == 5

    # Neighboring windows overlap by two slices, matching the Kaggle builder.
    np.testing.assert_array_equal(result.images[0, 0, 1], result.images[0, 1, 0])
    np.testing.assert_array_equal(result.images[0, 0, 2], result.images[0, 1, 1])


def test_window_with_two_valid_slices_is_kept_and_position_ignores_invalid():
    arrays = {role: _role_arrays() for role in FIVE_ROLE_ORDER}
    slices, valid, positions, info = arrays["SAG_FLUID"]
    valid[1] = 0
    positions[1] = 0.99  # This invalid slice must not contribute to window position.
    arrays["SAG_FLUID"] = (slices, valid, positions, info)

    result = assemble_five_role_windows(
        arrays,
        study_uid="study-1",
        role_series_uids={role: role for role in FIVE_ROLE_ORDER},
    )

    # Window [0, 1, 2] has two valid slices, therefore it remains usable.
    assert result.window_valid[0, 0] == 1
    expected = float(np.mean([positions[0], positions[2]]))
    assert np.isclose(result.window_positions[0, 0], expected)


def test_missing_role_is_zero_padded_and_reported():
    arrays = {role: _role_arrays() for role in FIVE_ROLE_ORDER if role != "AX_FLUID"}
    result = assemble_five_role_windows(
        arrays,
        study_uid="study-1",
        role_series_uids={role: "" if role == "AX_FLUID" else role for role in FIVE_ROLE_ORDER},
    )

    ax_index = FIVE_ROLE_ORDER.index("AX_FLUID")
    assert result.window_valid[ax_index].sum() == 0
    assert not result.images[ax_index].any()
    assert result.diagnostics[ax_index]["status"] == "no_candidate_series"
