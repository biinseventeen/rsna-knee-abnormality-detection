from __future__ import annotations

from typing import Any

from .prompts import LABEL_STATUSES


def parse_label_json(
    payload: dict[str, Any],
    labels: list[str],
) -> dict[str, str]:
    missing = [label for label in labels if label not in payload]
    if missing:
        raise ValueError(f"Missing labels: {missing}")

    invalid = {
        label: payload[label]
        for label in labels
        if payload[label] not in LABEL_STATUSES
    }
    if invalid:
        raise ValueError(f"Invalid label statuses: {invalid}")

    return {label: payload[label] for label in labels}
