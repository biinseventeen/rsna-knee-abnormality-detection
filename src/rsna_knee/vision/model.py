from __future__ import annotations

import torch
from torch import nn


class MultiLabelHead(nn.Module):
    def __init__(self, feature_dim: int = 384, n_labels: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_labels),
        )

    def forward(self, features):
        return self.net(features)


def load_head(
    checkpoint,
    feature_dim: int = 384,
    n_labels: int = 12,
    device="cpu",
):
    model = MultiLabelHead(feature_dim=feature_dim, n_labels=n_labels)
    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    model.load_state_dict(payload)
    model.eval().to(device)
    return model
