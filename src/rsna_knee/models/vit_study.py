"""Exact ViT-B/16 Residual Attention Pooling study model from Kaggle.

Source of truth:
    08_model_ensemble_kaggle_dataset_ready.ipynb, Cell 17.

The module names, parameter names, registered buffers and layer dimensions are
preserved for strict loading of vit_pretrained_best_checkpoint.pt.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ViTModelConfig:
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 256
    max_slices: int = 36
    study_depth: int = 2
    study_heads: int = 4
    freeze_early_blocks: int = 10
    slice_chunk_size: int = 8
    use_center_prior: bool = True
    center_prior_strength: float = 0.15
    center_prior_sigma: float = 0.55

    @classmethod
    def from_run_config(cls, run_config: dict) -> "ViTModelConfig":
        return cls(
            image_size=int(run_config.get("image_size", 224)),
            patch_size=int(run_config.get("patch_size", 16)),
            embed_dim=int(run_config.get("embed_dim", 256)),
            max_slices=int(run_config.get("max_slices", 36)),
            study_depth=int(run_config.get("study_depth", 2)),
            study_heads=int(run_config.get("study_heads", 4)),
            freeze_early_blocks=int(run_config.get("freeze_early_blocks", 10)),
            slice_chunk_size=int(run_config.get("slice_chunk_size", 8)),
            use_center_prior=bool(run_config.get("use_center_prior", True)),
            center_prior_strength=float(run_config.get("center_prior_strength", 0.15)),
            center_prior_sigma=float(run_config.get("center_prior_sigma", 0.55)),
        )

    def validate(self) -> None:
        if self.image_size != 224 or self.patch_size != 16:
            raise ValueError(
                f"Expected ViT-B/16 224px input; got image_size={self.image_size}, "
                f"patch_size={self.patch_size}."
            )
        if self.max_slices != 36:
            raise ValueError(f"Expected max_slices=36, got {self.max_slices}.")
        if self.embed_dim != 256:
            raise ValueError(f"Expected embed_dim=256, got {self.embed_dim}.")
        if self.study_depth != 2 or self.study_heads != 4:
            raise ValueError(
                f"Unexpected study transformer config: depth={self.study_depth}, "
                f"heads={self.study_heads}."
            )
        if not 0 <= self.freeze_early_blocks < 12:
            raise ValueError("freeze_early_blocks must be in [0, 11].")
        if self.slice_chunk_size < 1:
            raise ValueError("slice_chunk_size must be >= 1.")
        if self.center_prior_sigma <= 0:
            raise ValueError("center_prior_sigma must be positive.")


class PretrainedSliceEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        output_dim: int = 256,
        freeze_early_blocks: int = 10,
        use_center_prior: bool = True,
        center_prior_strength: float = 0.15,
        center_prior_sigma: float = 0.55,
        dropout: float = 0.10,
        image_size: int = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = 768
        self.freeze_early_blocks = int(freeze_early_blocks)
        self.use_center_prior = bool(use_center_prior)

        blocks = list(self.backbone.encoder.layers.children())
        if not 0 <= self.freeze_early_blocks < len(blocks):
            raise ValueError(
                f"freeze_early_blocks must be in [0, {len(blocks)-1}]"
            )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in blocks[self.freeze_early_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.backbone.encoder.ln.parameters():
            parameter.requires_grad = True

        self.trainable_blocks = blocks[self.freeze_early_blocks:]
        self.patch_attention = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        grid = image_size // patch_size
        coords = torch.linspace(-1.0, 1.0, grid)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        sigma = max(float(center_prior_sigma), 1e-4)
        log_gaussian = -0.5 * ((xx / sigma) ** 2 + (yy / sigma) ** 2)
        log_gaussian = log_gaussian - log_gaussian.max()
        strength = float(center_prior_strength) if use_center_prior else 0.0
        self.register_buffer(
            "center_bias",
            (strength * log_gaussian.reshape(-1)).float(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        for block in self.trainable_blocks:
            block.train(mode)
        self.backbone.encoder.ln.train(mode)
        self.patch_attention.train(mode)
        self.feature_projection.train(mode)
        return self

    def forward(self, images_rgb: torch.Tensor) -> torch.Tensor:
        # images_rgb: [N, 3, 224, 224], already ImageNet-normalized.
        x = self.backbone._process_input(images_rgb)
        cls = self.backbone.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        x = self.backbone.encoder.dropout(x)

        for block_i, block in enumerate(self.backbone.encoder.layers.children()):
            if block_i < self.freeze_early_blocks:
                with torch.no_grad():
                    x = block(x)
            else:
                x = block(x)

        x = self.backbone.encoder.ln(x)
        cls_feature = x[:, 0]
        patch_tokens = x[:, 1:]

        patch_scores = self.patch_attention(patch_tokens).squeeze(-1)
        patch_scores = patch_scores + self.center_bias.unsqueeze(0)
        patch_weights = torch.softmax(patch_scores, dim=1)
        spatial_feature = torch.sum(
            patch_tokens * patch_weights.unsqueeze(-1), dim=1
        )

        return self.feature_projection(
            torch.cat([cls_feature, spatial_feature], dim=-1)
        )


class StudyFeatureAggregator(nn.Module):
    """Gated metadata + study Transformer + residual mean/attention pooling."""

    def __init__(
        self,
        embed_dim: int = 256,
        max_slices: int = 36,
        study_depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.plane_embedding = nn.Embedding(4, embed_dim, padding_idx=3)
        self.metadata_projection = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.slice_order_embedding = nn.Embedding(max_slices, embed_dim)
        self.metadata_scale = nn.Parameter(torch.tensor(0.10))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.study_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=study_depth,
            enable_nested_tensor=False,
        )

        self.attention_score = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip_scale = nn.Parameter(torch.tensor(0.5))
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        slice_features: torch.Tensor,
        valid: torch.Tensor,
        plane_ids: torch.Tensor,
        fluid: torch.Tensor,
        fat: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_slices, embed_dim = slice_features.shape
        valid = valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Every study must contain at least one valid slice.")

        plane_ids = plane_ids.clamp(0, 3)
        metadata = torch.stack(
            [fluid.clamp(-1, 1), fat.clamp(-1, 1), positions.clamp(0, 1)],
            dim=-1,
        )
        order_ids = torch.arange(
            num_slices, device=slice_features.device
        ).unsqueeze(0)
        metadata_features = (
            self.plane_embedding(plane_ids)
            + self.metadata_projection(metadata)
            + self.slice_order_embedding(order_ids)
        )
        tokens = slice_features + self.metadata_scale * metadata_features

        encoded = self.study_encoder(tokens, src_key_padding_mask=~valid)
        mask = valid.unsqueeze(-1).to(encoded.dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        mean_pool = (encoded * mask).sum(dim=1) / denominator

        scores = self.attention_score(encoded).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attention_weights = torch.softmax(scores, dim=1)
        attention_pool = torch.sum(
            encoded * attention_weights.unsqueeze(-1), dim=1
        )
        pooled = self.pool_projection(torch.cat([mean_pool, attention_pool], dim=-1))
        raw_mean = (slice_features * mask).sum(dim=1) / denominator
        return self.output_norm(pooled + self.skip_scale * raw_mean)


class PretrainedViTStudyModel(nn.Module):
    """Final Residual Attention Pooling ViT-B/16 study-level classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        config: ViTModelConfig | None = None,
        num_targets: int = len(TARGETS),
    ) -> None:
        super().__init__()
        self.config = config or ViTModelConfig()
        self.config.validate()
        self.slice_chunk_size = int(self.config.slice_chunk_size)
        self.slice_encoder = PretrainedSliceEncoder(
            backbone=backbone,
            output_dim=self.config.embed_dim,
            freeze_early_blocks=self.config.freeze_early_blocks,
            use_center_prior=self.config.use_center_prior,
            center_prior_strength=self.config.center_prior_strength,
            center_prior_sigma=self.config.center_prior_sigma,
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
        )
        self.aggregator = StudyFeatureAggregator(
            embed_dim=self.config.embed_dim,
            max_slices=self.config.max_slices,
            study_depth=self.config.study_depth,
            num_heads=self.config.study_heads,
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.config.embed_dim, 128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_targets),
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def forward(
        self,
        images: torch.Tensor,
        valid: torch.Tensor,
        plane_ids: torch.Tensor,
        fluid: torch.Tensor,
        fat: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_slices, height, width = images.shape
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"Expected {self.config.image_size}x{self.config.image_size}; "
                f"got {height}x{width}"
            )
        if num_slices != self.config.max_slices:
            raise ValueError(
                f"Expected {self.config.max_slices} slices; got {num_slices}."
            )

        valid_bool = valid.bool()
        valid_flat = valid_bool.reshape(-1)
        valid_indices = valid_flat.nonzero(as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            raise ValueError("Batch contains no valid MRI slices")

        flat = images.reshape(batch_size * num_slices, 1, height, width)
        selected = flat.index_select(0, valid_indices).float().div(255.0)
        selected = selected.repeat(1, 3, 1, 1)
        selected = (selected - self.image_mean) / self.image_std

        encoded_chunks = []
        for start in range(0, selected.shape[0], self.slice_chunk_size):
            chunk = selected[start : start + self.slice_chunk_size]
            encoded_chunks.append(self.slice_encoder(chunk))
        valid_features = torch.cat(encoded_chunks, dim=0)

        flat_features = valid_features.new_zeros(
            (batch_size * num_slices, self.config.embed_dim)
        )
        flat_features = flat_features.index_copy(0, valid_indices, valid_features)
        slice_features = flat_features.view(
            batch_size, num_slices, self.config.embed_dim
        )

        study_features = self.aggregator(
            slice_features,
            valid_bool,
            plane_ids,
            fluid,
            fat,
            positions,
        )
        return self.classifier(study_features)
