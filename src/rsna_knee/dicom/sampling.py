from __future__ import annotations

from pathlib import Path


def select_center_triplet(paths: list[Path]) -> list[Path]:
    if not paths:
        raise ValueError("No DICOM slices available.")

    n = len(paths)
    if n <= 3:
        selected = list(paths)
    else:
        center = n // 2
        selected = paths[max(0, center - 1): center + 2]

    while len(selected) < 3:
        selected.append(selected[-1])

    return selected[:3]
