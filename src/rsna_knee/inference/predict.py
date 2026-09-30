from __future__ import annotations

import json
from pathlib import Path

import torch

from ..config import DINO_MODEL_ID, InferenceConfig
from ..dicom.io import find_dicom_files
from ..dicom.preprocess import make_rgb_stack
from ..dicom.sampling import select_center_triplet
from ..dicom.series import group_by_series, sort_series
from ..vision.aggregation import mean_pool
from ..vision.backbone import extract_feature, load_dinov2
from ..vision.model import load_head


def predict_from_paths(
    inputs: list[Path],
    checkpoint: Path,
    device: str = "auto",
    labels: list[str] | None = None,
) -> dict:
    labels = labels or []
    cfg = InferenceConfig(device=device)

    paths = find_dicom_files(inputs)
    if not paths:
        raise ValueError("No DICOM files found.")

    processor, backbone, resolved_device = load_dinov2(device)
    head = load_head(
        checkpoint,
        feature_dim=cfg.feature_dim,
        n_labels=len(labels),
        device=resolved_device,
    )

    series_groups = group_by_series(paths)
    series_features = []

    for _, series_paths in series_groups.items():
        series_paths = sort_series(series_paths)
        triplet = select_center_triplet(series_paths)
        image = make_rgb_stack(triplet, image_size=cfg.image_size)
        feature = extract_feature(
            image, processor, backbone, resolved_device
        )
        series_features.append(feature.squeeze(0))

    study_feature = mean_pool(series_features).unsqueeze(0)
    logits = head(study_feature)
    probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    return {
        "model": {
            "backbone": DINO_MODEL_ID,
            "checkpoint": str(checkpoint),
            "device": str(resolved_device),
        },
        "input": {
            "dicom_files": len(paths),
            "series": len(series_groups),
        },
        "predictions": {
            label: float(probability)
            for label, probability in zip(labels, probabilities)
        },
    }


def save_prediction(prediction: dict, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(prediction, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
