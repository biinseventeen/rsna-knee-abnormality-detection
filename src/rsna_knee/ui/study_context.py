"""Human-readable study context for the local RSNA knee demo dashboard.

Only non-identifying DICOM fields are read. PatientName and PatientID are
intentionally never loaded or displayed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class PatientInfo:
    age: str = "Không có dữ liệu"
    sex: str = "Không có dữ liệu"
    knee_side: str = "Không có dữ liệu"
    study_date: str = "Không có dữ liệu"
    body_part: str = "Không có dữ liệu"
    modality: str = "Không có dữ liệu"


@dataclass(frozen=True)
class RadiologyReport:
    text: str
    source: str
    path: Path
    uid_verified: bool


class StudyContextError(RuntimeError):
    """Raised when a report is unreadable or belongs to another study."""


REPORT_CANDIDATES = (
    "radiology_report.txt",
    "radiology_report.md",
    "doctor_report.txt",
    "doctor_report.md",
)


def _clean_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "none", "nan", "null", "n/a", "na"}:
        return ""
    return text


def format_patient_age(value: Any) -> str:
    raw = _clean_value(value).upper()
    if not raw:
        return "Không có dữ liệu"
    match = re.fullmatch(r"(\d{1,3})\s*([YMDW]?)", raw)
    if not match:
        return raw
    number = int(match.group(1))
    unit = match.group(2) or "Y"
    labels = {"Y": "tuổi", "M": "tháng", "W": "tuần", "D": "ngày"}
    return f"{number} {labels[unit]}"


def format_patient_sex(value: Any) -> str:
    raw = _clean_value(value).upper()
    return {
        "M": "Nam",
        "MALE": "Nam",
        "F": "Nữ",
        "FEMALE": "Nữ",
        "O": "Khác",
        "OTHER": "Khác",
        "U": "Không xác định",
        "UNKNOWN": "Không xác định",
    }.get(raw, "Không có dữ liệu")


def format_knee_side(value: Any) -> str:
    raw = _clean_value(value).upper()
    return {
        "L": "Gối trái",
        "LEFT": "Gối trái",
        "R": "Gối phải",
        "RIGHT": "Gối phải",
        "B": "Hai bên",
        "BILATERAL": "Hai bên",
    }.get(raw, "Không có dữ liệu")


def format_study_date(value: Any) -> str:
    raw = _clean_value(value)
    if not raw:
        return "Không có dữ liệu"
    digits = re.sub(r"[^0-9]", "", raw)
    if len(digits) >= 8:
        year, month, day = digits[:4], digits[4:6], digits[6:8]
        if year != "0000" and month != "00" and day != "00":
            return f"{day}/{month}/{year}"
    return raw


def patient_info_from_tags(
    tags: dict[str, Any] | Any,
    *,
    fallback_sex: Any = None,
) -> PatientInfo:
    """Convert a DICOM-like tag mapping into Vietnamese display fields."""
    getter = tags.get if hasattr(tags, "get") else lambda key, default=None: default
    sex_value = getter("PatientSex", None)
    if not _clean_value(sex_value):
        sex_value = fallback_sex

    side = getter("Laterality", None)
    if not _clean_value(side):
        side = getter("ImageLaterality", None)

    body_part = _clean_value(getter("BodyPartExamined", None))
    if body_part.upper() in {"KNEE", "KNEES"}:
        body_part = "Khớp gối"

    modality = _clean_value(getter("Modality", None))
    if modality.upper() == "MR":
        modality = "MRI"

    return PatientInfo(
        age=format_patient_age(getter("PatientAge", None)),
        sex=format_patient_sex(sex_value),
        knee_side=format_knee_side(side),
        study_date=format_study_date(getter("StudyDate", None)),
        body_part=body_part or "Không có dữ liệu",
        modality=modality or "Không có dữ liệu",
    )


def read_patient_info(study: Any) -> PatientInfo:
    """Read a few non-identifying tags from one or more representative slices."""
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - local optional dependency
        raise RuntimeError(
            "Thiếu pydicom. Cài bằng `python -m pip install -r requirements-ui.txt`."
        ) from exc

    specific_tags = [
        "PatientAge",
        "PatientSex",
        "StudyDate",
        "Laterality",
        "ImageLaterality",
        "BodyPartExamined",
        "Modality",
    ]
    collected: dict[str, Any] = {}
    # Read one valid header from up to three distinct series. Do not read any
    # direct patient identifiers such as PatientName or PatientID.
    for series in study.series.values():
        probe = next((item for item in series.files if not item.header_error), None)
        if probe is None:
            continue
        try:
            dataset = pydicom.dcmread(
                str(probe.path),
                stop_before_pixels=True,
                force=True,
                specific_tags=specific_tags,
            )
        except Exception:
            continue
        for key in specific_tags:
            if key not in collected and _clean_value(getattr(dataset, key, None)):
                collected[key] = getattr(dataset, key)
        if all(key in collected for key in ("PatientAge", "PatientSex", "Laterality", "StudyDate")):
            break
        if len(collected) >= 5:
            break

    clinical_context = study.manifest.get("clinical_context", {})
    if not isinstance(clinical_context, dict):
        clinical_context = {}
    fallback_sex = clinical_context.get("patient_sex_from_train_csv")
    return patient_info_from_tags(collected, fallback_sex=fallback_sex)


def _normalize_report_body(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        line = line.replace("\t", "    ").strip()
        blank = not line
        if blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = blank
    return "\n".join(normalized).strip()


def _split_report_header(text: str) -> tuple[str, str, bool]:
    """Read optional UID/source header written by the report-export notebook."""
    lines = text.splitlines()
    uid = ""
    source = "Báo cáo văn bản đi kèm sample"
    consumed = 0
    for line in lines[:4]:
        stripped = line.strip()
        if stripped.casefold().startswith("studyinstanceuid:"):
            uid = stripped.split(":", 1)[1].strip()
            consumed += 1
        elif stripped.casefold().startswith("source:"):
            source = stripped.split(":", 1)[1].strip() or source
            consumed += 1
        elif stripped in {"---", "----------------------------------------"}:
            consumed += 1
        elif not stripped and consumed:
            consumed += 1
        else:
            break
    if consumed:
        body = "\n".join(lines[consumed:])
        return body, source, bool(uid)
    return text, source, False


def load_radiology_report(
    sample_root: str | Path,
    expected_study_uid: str,
    *,
    report_path: str | Path | None = None,
) -> RadiologyReport | None:
    """Load a plain-text report, validating its optional StudyInstanceUID header."""
    root = Path(sample_root).expanduser().resolve()
    candidates: list[Path] = []
    if report_path is not None and str(report_path).strip():
        candidates.append(Path(report_path).expanduser().resolve())
    else:
        candidates.extend(root / name for name in REPORT_CANDIDATES)

    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return None

    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            raw = path.read_text(encoding="cp1252")
        except (OSError, UnicodeError) as exc:
            raise StudyContextError(f"Không đọc được báo cáo {path}: {exc}") from exc
    except OSError as exc:
        raise StudyContextError(f"Không đọc được báo cáo {path}: {exc}") from exc

    body, source, has_uid_header = _split_report_header(raw)
    uid_match = re.search(r"(?im)^StudyInstanceUID\s*:\s*(.+?)\s*$", raw)
    if uid_match:
        report_uid = uid_match.group(1).strip()
        if report_uid != str(expected_study_uid).strip():
            raise StudyContextError(
                "Báo cáo bác sĩ không thuộc study đang mở: "
                f"report={report_uid!r}, DICOM/prediction={expected_study_uid!r}."
            )
    normalized = _normalize_report_body(body)
    if not normalized:
        raise StudyContextError("File báo cáo tồn tại nhưng không có nội dung.")
    return RadiologyReport(
        text=normalized,
        source=source,
        path=path,
        uid_verified=bool(uid_match or has_uid_header),
    )
