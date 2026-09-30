from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .io import load_dicom


def group_by_series(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        ds = load_dicom(path)
        uid = str(getattr(ds, "SeriesInstanceUID", "UNKNOWN"))
        groups[uid].append(path)
    return dict(groups)


def _sort_key(path: Path):
    ds = load_dicom(path)

    instance = getattr(ds, "InstanceNumber", None)
    if instance is not None:
        return (0, int(instance))

    position = getattr(ds, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        return (1, float(position[2]))

    return (2, path.name)


def sort_series(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=_sort_key)
