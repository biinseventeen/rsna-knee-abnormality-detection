import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.ui.study_context import (  # noqa: E402
    StudyContextError,
    format_patient_age,
    format_patient_sex,
    format_knee_side,
    format_study_date,
    load_radiology_report,
    patient_info_from_tags,
)


def test_patient_metadata_is_converted_to_vietnamese_display_values():
    info = patient_info_from_tags(
        {
            "PatientAge": "056Y",
            "PatientSex": "F",
            "Laterality": "L",
            "StudyDate": "20240209",
            "BodyPartExamined": "KNEE",
            "Modality": "MR",
        }
    )
    assert info.age == "56 tuổi"
    assert info.sex == "Nữ"
    assert info.knee_side == "Gối trái"
    assert info.study_date == "09/02/2024"
    assert info.body_part == "Khớp gối"
    assert info.modality == "MRI"


def test_missing_patient_fields_are_not_invented():
    info = patient_info_from_tags({})
    assert info.age == "Không có dữ liệu"
    assert info.sex == "Không có dữ liệu"
    assert info.study_date == "Không có dữ liệu"
    assert info.knee_side == "Không có dữ liệu"


def test_train_csv_sex_can_be_used_only_as_fallback():
    info = patient_info_from_tags({"PatientSex": "", "BodyPartExamined": "KNEE"}, fallback_sex="Male")
    assert info.sex == "Nam"


def test_age_sex_side_and_date_formatters():
    assert format_patient_age("006M") == "6 tháng"
    assert format_patient_sex("O") == "Khác"
    assert format_knee_side("R") == "Gối phải"
    assert format_study_date("202411") == "202411"


def test_loads_and_formats_report_with_matching_study_uid(tmp_path):
    path = tmp_path / "radiology_report.txt"
    path.write_text(
        "StudyInstanceUID: study-123\n"
        "Source: train.csv / Report\n"
        "\n---\n"
        "FINDINGS:\r\nMild joint effusion.\r\n\r\n\r\nNo fracture.\r\n",
        encoding="utf-8",
    )
    report = load_radiology_report(tmp_path, "study-123")
    assert report is not None
    assert report.uid_verified is True
    assert report.source == "train.csv / Report"
    assert report.text == "FINDINGS:\nMild joint effusion.\n\nNo fracture."


def test_rejects_report_from_another_study(tmp_path):
    (tmp_path / "radiology_report.txt").write_text(
        "StudyInstanceUID: study-other\nSource: train.csv / Report\n\nText",
        encoding="utf-8",
    )
    with pytest.raises(StudyContextError, match="không thuộc study"):
        load_radiology_report(tmp_path, "study-123")


def test_missing_report_returns_none(tmp_path):
    assert load_radiology_report(tmp_path, "study-123") is None
