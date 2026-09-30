from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pydicom


def find_dicom_files(inputs: Iterable[Path]) -> list[Path]:
    files: list[Path] = []

    for item in inputs:
        item = Path(item)
        if item.is_dir():
            for p in item.rglob("*"):
                if p.is_file() and (p.suffix.lower() in {".dcm", ""}):
                    files.append(p)
        elif item.is_file():
            files.append(item)

    return sorted(set(files))


def load_dicom(path: Path):
    return pydicom.dcmread(str(path), force=True)


def load_dicom_header(path: Path) -> dict:
    ds = load_dicom(path)
    keys = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "InstanceNumber",
        "Rows",
        "Columns",
        "Modality",
        "SeriesDescription",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "PhotometricInterpretation",
        "RescaleSlope",
        "RescaleIntercept",
    ]
    result = {}
    for key in keys:
        value = getattr(ds, key, None)
        if hasattr(value, "tolist"):
            value = value.tolist()
        result[key] = value
    return result
