from __future__ import annotations

import torch


def mean_pool(features: list[torch.Tensor]) -> torch.Tensor:
    if not features:
        raise ValueError("No features to aggregate.")
    return torch.stack(features, dim=0).mean(dim=0)
