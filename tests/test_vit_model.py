import sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest
import torch

try:
    from torchvision.models import vit_b_16
except Exception as exc:  # pragma: no cover
    pytest.skip(f"torchvision ViT-B/16 unavailable: {exc}", allow_module_level=True)

from rsna_knee.models.vit_study import PretrainedViTStudyModel, ViTModelConfig


def test_vit_model_strict_state_dict_round_trip_and_forward():
    torch.set_num_threads(2)
    config = ViTModelConfig(slice_chunk_size=1)
    first = PretrainedViTStudyModel(vit_b_16(weights=None), config=config).eval()
    state = first.state_dict()
    second = PretrainedViTStudyModel(vit_b_16(weights=None), config=config).eval()
    incompatible = second.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    images = torch.zeros((1, 36, 224, 224), dtype=torch.uint8)
    valid = torch.zeros((1, 36), dtype=torch.bool)
    valid[0, 0] = True
    plane_ids = torch.full((1, 36), -1, dtype=torch.long)
    plane_ids[0, 0] = 0
    fluid = torch.full((1, 36), -1, dtype=torch.long)
    fat = torch.full((1, 36), -1, dtype=torch.long)
    positions = torch.zeros((1, 36), dtype=torch.float32)

    with torch.inference_mode():
        logits = second(images, valid, plane_ids, fluid, fat, positions)
        probabilities = torch.sigmoid(logits.float())

    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
