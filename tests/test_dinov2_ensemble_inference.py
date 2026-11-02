from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow tests to run directly from the project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import pytest
import torch
from torch import nn

from rsna_knee.inference.dinov2_ensemble import (
    DinoInferenceError,
    run_dinov2_five_fold_inference,
    save_dinov2_inference_result,
    validate_dinov2_inputs,
)
from rsna_knee.models.dinov2_five_role import TARGETS


class ConstantLogitModel(nn.Module):
    def __init__(self, logit: float):
        super().__init__()
        self.logit = float(logit)

    def forward(self, images, window_valid, window_positions):
        assert tuple(images.shape) == (1, 5, 10, 3, 224, 224)
        assert tuple(window_valid.shape) == (1, 5, 10)
        assert tuple(window_positions.shape) == (1, 5, 10)
        return torch.full((1, len(TARGETS)), self.logit, dtype=torch.float32, device=images.device)


def make_inputs(valid_count: int = 5):
    images = np.zeros((5, 10, 3, 224, 224), dtype=np.uint8)
    valid = np.zeros((5, 10), dtype=np.uint8)
    positions = np.zeros((5, 10), dtype=np.float32)
    # Spread valid windows across roles for the fake-model test.
    for i in range(valid_count):
        valid[i % 5, i // 5] = 1
        positions[i % 5, i // 5] = 0.25 + 0.1 * (i // 5)
    return images, valid, positions


def test_validate_inputs_keeps_uint8_and_expected_shapes():
    inputs = make_inputs()
    images, valid, positions = validate_dinov2_inputs(*inputs)
    assert images.dtype == np.uint8
    assert images.shape == (5, 10, 3, 224, 224)
    assert valid.dtype == np.bool_
    assert positions.dtype == np.float32


def test_rejects_float_images_to_prevent_double_normalization():
    images, valid, positions = make_inputs()
    with pytest.raises(DinoInferenceError, match="uint8"):
        validate_dinov2_inputs(images.astype(np.float32), valid, positions)


def test_five_fold_mean_is_average_of_sigmoid_probabilities():
    images, valid, positions = make_inputs(valid_count=5)
    logits_by_fold = [-2.0, -1.0, 0.0, 1.0, 2.0]
    loaded_folds = []

    def fake_loader(**kwargs):
        fold = kwargs["fold"]
        loaded_folds.append(fold)
        return ConstantLogitModel(logits_by_fold[fold - 1]), {
            "fold": fold,
            "strict_load": True,
            "best_epoch": 10 + fold,
        }

    result = run_dinov2_five_fold_inference(
        images=images,
        window_valid=valid,
        window_positions=positions,
        bundle_root="not-used-by-fake-loader",
        study_uid="sample-study",
        device="cpu",
        encoder_chunk_size=1,
        fold_loader=fake_loader,
        verbose=False,
    )

    expected_each = torch.sigmoid(torch.tensor(logits_by_fold)).numpy()
    expected_mean = float(expected_each.mean())
    assert loaded_folds == [1, 2, 3, 4, 5]
    assert result.status == "ok"
    assert result.fold_probabilities.shape == (5, 12)
    assert result.ensemble_probabilities.shape == (12,)
    np.testing.assert_allclose(result.fold_probabilities[:, 0], expected_each, rtol=1e-6)
    np.testing.assert_allclose(result.ensemble_probabilities, expected_mean, rtol=1e-6)
    assert len(result.fold_info) == 5


def test_all_invalid_windows_match_neutral_kaggle_fallback_without_loading_models():
    images, valid, positions = make_inputs(valid_count=0)

    def must_not_load(**kwargs):
        raise AssertionError("No model should be loaded for an all-invalid study.")

    result = run_dinov2_five_fold_inference(
        images=images,
        window_valid=valid,
        window_positions=positions,
        bundle_root="not-used",
        study_uid="empty-study",
        device="cpu",
        fold_loader=must_not_load,
        verbose=False,
    )
    assert result.status == "all_roles_unusable_fallback_0.5"
    np.testing.assert_array_equal(result.fold_probabilities, np.full((5, 12), 0.5, np.float32))
    np.testing.assert_array_equal(result.ensemble_probabilities, np.full(12, 0.5, np.float32))


def test_save_outputs_produces_auditable_files(tmp_path):
    images, valid, positions = make_inputs(valid_count=5)

    def fake_loader(**kwargs):
        fold = kwargs["fold"]
        return ConstantLogitModel(float(fold - 3)), {
            "fold": fold,
            "strict_load": True,
            "best_epoch": 5,
            "best_val_loss": 0.1,
        }

    result = run_dinov2_five_fold_inference(
        images=images,
        window_valid=valid,
        window_positions=positions,
        bundle_root="not-used",
        study_uid="test-uid",
        device="cpu",
        fold_loader=fake_loader,
        verbose=False,
    )
    paths = save_dinov2_inference_result(result, tmp_path)
    assert set(paths) == {
        "fold_probabilities_csv",
        "ensemble_prediction_csv",
        "predictions_npz",
        "report_json",
    }
    assert all(path.is_file() for path in paths.values())
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["aggregation"] == "mean of per-fold sigmoid probabilities"
    assert len(report["fold_info"]) == 5
