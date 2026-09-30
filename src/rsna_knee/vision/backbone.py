from __future__ import annotations

import torch
from transformers import AutoImageProcessor, AutoModel

from ..config import DINO_MODEL_ID


def resolve_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dinov2(device: str = "auto"):
    resolved = resolve_device(device)
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModel.from_pretrained(DINO_MODEL_ID)
    model.eval().to(resolved)
    return processor, model, resolved


@torch.inference_mode()
def extract_feature(image, processor, model, device: torch.device):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0]
