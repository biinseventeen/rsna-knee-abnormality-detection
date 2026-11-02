from pathlib import Path

import numpy as np
import pytest
import torch

from rsna_knee.inference.vit_full_study import (
    ViTInferenceError,
    predict_vit_full_study,
    save_vit_full_study_outputs,
    validate_vit_inputs,
)
from rsna_knee.models.vit_study import TARGETS


class DeterministicFakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.last_valid_count = None
        self.last_args = None

    def forward(self, images, valid, plane_ids, fluid, fat, positions):
        self.last_args = (images, valid, plane_ids, fluid, fat, positions)
        self.last_valid_count = int(valid.bool().sum().item())
        score = (
            valid.float().sum(dim=1, keepdim=True) * 0.01
            + plane_ids.float().clamp_min(0).sum(dim=1, keepdim=True) * 0.001
        )
        offsets = torch.arange(len(TARGETS), device=images.device).float().view(1, -1) / 10
        return score + offsets + self.anchor


def make_payload(n_valid=9):
    images = np.zeros((36, 224, 224), dtype=np.uint8)
    valid = np.zeros(36, dtype=np.uint8)
    valid[:n_valid] = 1
    plane_ids = np.full(36, -1, dtype=np.int8)
    plane_ids[:n_valid] = np.arange(n_valid, dtype=np.int8) % 3
    fluid = np.full(36, -1, dtype=np.int8)
    fat = np.full(36, -1, dtype=np.int8)
    fluid[:n_valid] = 1
    fat[:n_valid] = 0
    positions = np.zeros(36, dtype=np.float32)
    positions[:n_valid] = np.linspace(0, 1, n_valid, dtype=np.float32)
    return {
        "images": images,
        "valid": valid,
        "plane_ids": plane_ids,
        "fluid": fluid,
        "fat": fat,
        "positions": positions,
    }


def test_input_contract_accepts_full_study():
    payload = validate_vit_inputs(make_payload())
    assert payload["images"].shape == (36, 224, 224)
    assert payload["valid"].sum() == 9


def test_rejects_no_valid_slices():
    with pytest.raises(ViTInferenceError, match="zero valid slices"):
        validate_vit_inputs(make_payload(n_valid=0))


def test_full_valid_mask_and_forward_order_are_preserved():
    payload = make_payload(n_valid=17)
    model = DeterministicFakeModel()
    result = predict_vit_full_study(
        model, payload, study_uid="study-x", device="cpu"
    )
    assert model.last_valid_count == 17
    assert tuple(model.last_args[0].shape) == (1, 36, 224, 224)
    assert tuple(model.last_args[1].shape) == (1, 36)
    assert result.logits.shape == (12,)
    assert result.probabilities.shape == (12,)
    assert np.allclose(result.probabilities, 1 / (1 + np.exp(-result.logits)), atol=1e-6)


def test_probabilities_are_finite_and_bounded():
    result = predict_vit_full_study(
        DeterministicFakeModel(), make_payload(), study_uid="study-y", device="cpu"
    )
    assert np.isfinite(result.logits).all()
    assert np.isfinite(result.probabilities).all()
    assert ((result.probabilities >= 0) & (result.probabilities <= 1)).all()


def test_outputs_are_saved(tmp_path: Path):
    prediction = predict_vit_full_study(
        DeterministicFakeModel(), make_payload(), study_uid="study-z", device="cpu"
    )
    paths = save_vit_full_study_outputs(
        prediction,
        tmp_path,
        model_info={
            "strict_load": True,
            "parameter_count": 1,
            "architecture": "fake-for-test",
            "run_config": {},
        },
        restore_seconds=0.1,
        inference_seconds=0.2,
        cache_path=tmp_path / "cache.npz",
    )
    assert all(path.is_file() for path in paths.values())
    import pandas as pd
    frame = pd.read_csv(paths["csv"])
    assert len(frame) == 12
    assert frame["target"].tolist() == list(TARGETS)
    with np.load(paths["npz"], allow_pickle=False) as data:
        assert data["probabilities"].shape == (12,)
