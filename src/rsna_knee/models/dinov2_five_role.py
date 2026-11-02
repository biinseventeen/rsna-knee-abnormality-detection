"""Exact five-role 2.5D DINOv2 model used by the final Kaggle ensemble.

Source of truth:
    08_model_ensemble_kaggle_dataset_ready.ipynb
    Cell 29: "DINOv2: exact five-role V2 architecture"

Important compatibility note:
    The module names, parameter names, buffer names, layer dimensions and
    forward operations in ``FiveRoleDINOv2_2p5D`` are intentionally preserved
    so ``load_state_dict(..., strict=True)`` can validate the original fold
    checkpoints. Do not casually rename layers or add persistent buffers.
"""

from __future__ import annotations

import math
from typing import Final

import torch
import torch.nn as nn

TARGETS: Final[tuple[str, ...]] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)

FIVE_ROLE_ORDER: Final[tuple[str, ...]] = (
    "SAG_FLUID",
    "COR_FLUID",
    "AX_FLUID",
    "SAG_NONFLUID",
    "COR_NONFLUID",
)

IMAGE_SIZE: Final[int] = 224
PATCH_SIZE: Final[int] = 14
BACKBONE_DIM: Final[int] = 384
EMBED_DIM: Final[int] = 256
N_ROLE_SLICES: Final[int] = 12
N_ROLE_WINDOWS: Final[int] = N_ROLE_SLICES - 2
DINO_HUB_MODEL: Final[str] = "dinov2_vits14"


class FiveRoleDINOv2_2p5D(nn.Module):
    """Shared DINOv2 ViT-S/14 encoder with target-by-role study pooling.

    Input contract (same as the Kaggle final inference notebook):
        images:          uint8 [B, 5, 10, 3, 224, 224]
        window_valid:    bool/uint8 [B, 5, 10]
        window_positions: float [B, 5, 10], normalized to [0, 1]

    Output:
        logits: float [B, 12]. Apply sigmoid outside the model to obtain
        per-target probabilities.
    """

    def __init__(
        self,
        backbone: nn.Module,
        image_size: int = IMAGE_SIZE,
        backbone_dim: int = BACKBONE_DIM,
        embed_dim: int = EMBED_DIM,
        freeze_early_blocks: int = 9,
        dropout: float = 0.15,
        encoder_chunk_size: int = 8,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.image_size = int(image_size)
        self.backbone_dim = int(backbone_dim)
        self.embed_dim = int(embed_dim)
        self.encoder_chunk_size = int(encoder_chunk_size)
        self.n_roles = len(FIVE_ROLE_ORDER)
        self.n_targets = len(TARGETS)
        self.n_windows = N_ROLE_WINDOWS

        if self.image_size % PATCH_SIZE != 0:
            raise ValueError("Image size must be divisible by DINOv2 patch size 14.")
        if not hasattr(self.backbone, "blocks") or len(self.backbone.blocks) != 12:
            raise ValueError("Expected DINOv2 ViT-S/14 with 12 transformer blocks.")
        if self.n_roles != 5:
            raise ValueError(f"Expected exactly five roles, found {self.n_roles}.")
        if self.encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be >= 1.")

        # Preserve the original fine-tuning policy: freeze the backbone first,
        # then unfreeze its last blocks and final LayerNorm.
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if not 0 <= freeze_early_blocks < 12:
            raise ValueError("freeze_early_blocks must be between 0 and 11.")
        self.trainable_blocks = list(self.backbone.blocks[freeze_early_blocks:])
        for block in self.trainable_blocks:
            for p in block.parameters():
                p.requires_grad_(True)
        for p in self.backbone.norm.parameters():
            p.requires_grad_(True)

        # Convert each DINOv2 window into one compact feature vector.
        self.patch_attention = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, 1, bias=False),
        )
        self.window_projection = nn.Sequential(
            nn.Linear(self.backbone_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Mild spatial prior; peripheral patches remain available to attention.
        grid = self.image_size // PATCH_SIZE
        coords = torch.linspace(-1.0, 1.0, grid)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        log_gaussian = -0.5 * ((xx / 0.55) ** 2 + (yy / 0.55) ** 2)
        log_gaussian = log_gaussian - log_gaussian.max()
        self.register_buffer("center_bias", (0.15 * log_gaussian.reshape(-1)).float())

        self.role_embedding = nn.Parameter(
            torch.zeros(1, self.n_roles, 1, self.embed_dim)
        )
        self.window_embedding = nn.Embedding(self.n_windows, self.embed_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(3, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        nn.init.normal_(self.role_embedding, std=0.02)

        # Independent attention query for every (target, role) pair.
        self.target_role_queries = nn.Parameter(
            torch.empty(self.n_targets, self.n_roles, self.embed_dim)
        )
        nn.init.normal_(self.target_role_queries, std=0.5)

        # h(target, role) = global_role + gamma * (attended_role - global_role).
        gamma_init = 0.70
        gamma_logit = math.log(gamma_init / (1.0 - gamma_init))
        self.target_role_residual_logits = nn.Parameter(
            torch.full((self.n_targets, self.n_roles), gamma_logit)
        )
        self.role_feature_norm = nn.LayerNorm(self.embed_dim)

        # Every target gets its own prediction head. Each head sees the five
        # role-specific vectors concatenated in FIVE_ROLE_ORDER, plus presence
        # indicators so a missing role is distinguishable from a real zero.
        head_input_dim = self.n_roles * self.embed_dim + self.n_roles
        self.target_classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(head_input_dim),
                    nn.Linear(head_input_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.25),
                    nn.Linear(128, 1),
                )
                for _ in range(self.n_targets)
            ]
        )

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen DINO blocks remain in eval mode; selected final blocks train.
        self.backbone.eval()
        for block in self.trainable_blocks:
            block.train(mode)
        self.backbone.norm.train(mode)
        return self

    def forward(
        self,
        images: torch.Tensor,
        window_valid: torch.Tensor,
        window_positions: torch.Tensor,
    ) -> torch.Tensor:
        # images: [B, 5, 10, 3, 224, 224], uint8
        # masks/positions: [B, 5, 10]
        if images.ndim != 6:
            raise ValueError(f"Expected 6D images, received {tuple(images.shape)}")
        batch_size, roles, windows, channels, height, width = images.shape
        expected = (
            self.n_roles,
            self.n_windows,
            3,
            self.image_size,
            self.image_size,
        )
        if (roles, windows, channels, height, width) != expected:
            raise ValueError(
                f"Unexpected image shape {tuple(images.shape)}; expected [B, {expected}]."
            )
        if window_valid.shape != (batch_size, roles, windows):
            raise ValueError("window_valid shape does not match images.")
        if window_positions.shape != (batch_size, roles, windows):
            raise ValueError("window_positions shape does not match images.")
        if not window_valid.bool().any(dim=(1, 2)).all():
            raise ValueError("At least one study in the batch has no valid windows.")

        # Encode only valid windows; do not waste DINOv2 compute on padded slots.
        flat_valid = window_valid.bool().reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(1)
        flat_images = images.reshape(batch_size * roles * windows, channels, height, width)
        selected = flat_images.index_select(0, valid_indices).float().div_(255.0)
        selected = (selected - self.image_mean) / self.image_std

        encoded_chunks = []
        for start in range(0, selected.shape[0], self.encoder_chunk_size):
            chunk = selected[start : start + self.encoder_chunk_size]
            features = self.backbone.forward_features(chunk)
            cls_token = features["x_norm_clstoken"]
            patch_tokens = features["x_norm_patchtokens"]
            patch_scores = self.patch_attention(patch_tokens).squeeze(-1)
            patch_scores = patch_scores + self.center_bias.unsqueeze(0)
            patch_weights = torch.softmax(patch_scores, dim=1)
            patch_pool = torch.sum(
                patch_tokens * patch_weights.unsqueeze(-1), dim=1
            )
            encoded_chunks.append(
                self.window_projection(torch.cat([cls_token, patch_pool], dim=-1))
            )
        valid_features = torch.cat(encoded_chunks, dim=0)

        flat_features = valid_features.new_zeros(
            (batch_size * roles * windows, self.embed_dim)
        )
        flat_features = flat_features.index_copy(0, valid_indices, valid_features)
        tokens = flat_features.view(batch_size, roles, windows, self.embed_dim)

        # Add role, window-order and normalized slice-position information.
        pos = window_positions.to(dtype=tokens.dtype).clamp(0.0, 1.0)
        pos_features = torch.stack(
            [pos, torch.sin(2.0 * math.pi * pos), torch.cos(2.0 * math.pi * pos)],
            dim=-1,
        )
        pos_features = self.position_projection(pos_features)
        order_ids = torch.arange(windows, device=images.device)
        tokens = (
            tokens
            + self.role_embedding.to(dtype=tokens.dtype)
            + self.window_embedding(order_ids).view(1, 1, windows, self.embed_dim)
            + pos_features
        )

        # Per-role global means: [B, R, D]. Missing roles remain zero.
        valid = window_valid.bool()
        mask_f = valid.unsqueeze(-1).to(tokens.dtype)
        role_counts = mask_f.sum(dim=2).clamp_min(1.0)
        role_global = (tokens * mask_f).sum(dim=2) / role_counts

        # Independent attention over windows WITHIN each role for each target.
        # Scores: [B, T, R, W]; masked softmax safely handles absent roles.
        scores = torch.einsum(
            "brwd,trd->btrw", tokens, self.target_role_queries
        ) / math.sqrt(self.embed_dim)
        scores = scores.masked_fill(
            ~valid[:, None, :, :], torch.finfo(scores.dtype).min
        )
        attention = torch.softmax(scores.float(), dim=-1).to(tokens.dtype)
        attention = attention * valid[:, None, :, :].to(tokens.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attended_role = torch.einsum("btrw,brwd->btrd", attention, tokens)

        gamma = torch.sigmoid(self.target_role_residual_logits).view(
            1, self.n_targets, self.n_roles, 1
        )
        target_role_features = role_global[:, None, :, :] + gamma * (
            attended_role - role_global[:, None, :, :]
        )
        target_role_features = self.role_feature_norm(target_role_features)

        # Keep missing-role vectors at zero after LayerNorm; the presence flag
        # below explicitly tells the target head that the role was unavailable.
        role_presence = valid.any(dim=-1).to(target_role_features.dtype)  # [B, R]
        target_role_features = target_role_features * role_presence[
            :, None, :, None
        ]

        # Concatenate five independently pooled role vectors per target.
        target_role_features = target_role_features.reshape(
            batch_size, self.n_targets, self.n_roles * self.embed_dim
        )
        presence_features = role_presence[:, None, :].expand(
            -1, self.n_targets, -1
        )
        classifier_inputs = torch.cat(
            [target_role_features, presence_features], dim=-1
        )

        # Independent nonlinear prediction head for each target.
        logits = torch.cat(
            [
                self.target_classifiers[target](classifier_inputs[:, target, :])
                for target in range(self.n_targets)
            ],
            dim=1,
        )
        return logits
