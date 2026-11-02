from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsna_knee.models.dinov2_five_role import (
    BACKBONE_DIM,
    FiveRoleDINOv2_2p5D,
    IMAGE_SIZE,
    N_ROLE_WINDOWS,
)


class TinyTestBackbone(nn.Module):
    """Small deterministic stand-in for testing the outer model contract."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
        self.patch_projection = nn.Linear(3, BACKBONE_DIM)
        self.norm = nn.LayerNorm(BACKBONE_DIM)

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        patches = F.avg_pool2d(images, kernel_size=14, stride=14)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.norm(self.patch_projection(patches))
        cls_token = self.norm(patches.mean(dim=1))
        return {
            "x_norm_clstoken": cls_token,
            "x_norm_patchtokens": patches,
        }


def test_model_state_dict_strict_round_trip() -> None:
    model_a = FiveRoleDINOv2_2p5D(
        TinyTestBackbone(), encoder_chunk_size=2
    )
    model_b = FiveRoleDINOv2_2p5D(
        TinyTestBackbone(), encoder_chunk_size=2
    )
    result = model_b.load_state_dict(model_a.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_five_role_forward_output_shape_and_finiteness() -> None:
    model = FiveRoleDINOv2_2p5D(
        TinyTestBackbone(), encoder_chunk_size=2
    ).eval()

    # Preserve the full Kaggle input tensor layout but mark one window per role
    # valid so this unit test encodes only five windows, not all fifty.
    images = torch.randint(
        0, 256, (1, 5, N_ROLE_WINDOWS, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8
    )
    valid = torch.zeros((1, 5, N_ROLE_WINDOWS), dtype=torch.bool)
    valid[:, :, 4] = True
    positions = torch.zeros((1, 5, N_ROLE_WINDOWS), dtype=torch.float32)
    positions[:, :, 4] = 0.5

    with torch.inference_mode():
        logits = model(images, valid, positions)
        probabilities = torch.sigmoid(logits)

    assert logits.shape == (1, 12)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_model_rejects_study_with_no_valid_windows() -> None:
    model = FiveRoleDINOv2_2p5D(TinyTestBackbone()).eval()
    images = torch.zeros(
        (1, 5, N_ROLE_WINDOWS, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8
    )
    valid = torch.zeros((1, 5, N_ROLE_WINDOWS), dtype=torch.bool)
    positions = torch.zeros((1, 5, N_ROLE_WINDOWS), dtype=torch.float32)

    try:
        model(images, valid, positions)
    except ValueError as exc:
        assert "no valid windows" in str(exc)
    else:  # pragma: no cover - explicit test failure branch
        raise AssertionError("Expected ValueError for an entirely invalid study.")
