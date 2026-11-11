from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_demo_batch_inference.py"
SPEC = importlib.util.spec_from_file_location("run_demo_batch_inference_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def make_package(root: Path, folder: str, uid: str) -> Path:
    package = root / folder
    (package / "dicom" / uid / "series-a").mkdir(parents=True)
    (package / "series_metadata.csv").write_text("SeriesInstanceUID\nseries-a\n", encoding="utf-8")
    (package / "series_selection.csv").write_text("model,role_or_plane,SeriesInstanceUID\n", encoding="utf-8")
    (package / "manifest.json").write_text(
        json.dumps({"source": {"study_instance_uid": uid}}),
        encoding="utf-8",
    )
    return package


def test_discovers_nested_study_packages(tmp_path):
    a = make_package(tmp_path, "study_002/demo_sample", "1.2.3")
    b = make_package(tmp_path, "study_003", "1.2.4")
    found = MODULE.discover_study_packages(tmp_path)
    assert set(found) == {a.resolve(), b.resolve()}


def test_manifest_uid_matches_dicom_folder(tmp_path):
    package = make_package(tmp_path, "study_002", "1.2.3")
    assert MODULE._study_uid_from_package(package) == "1.2.3"


def test_manifest_uid_mismatch_is_rejected(tmp_path):
    package = make_package(tmp_path, "study_002", "1.2.3")
    manifest = {"source": {"study_instance_uid": "9.9.9"}}
    (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    try:
        MODULE._study_uid_from_package(package)
    except ValueError as exc:
        assert "differs from DICOM folder UID" in str(exc)
    else:
        raise AssertionError("Expected UID mismatch to be rejected")


def test_prediction_bundle_complete_requires_matching_uid(tmp_path):
    folder = tmp_path / "1.2.3"
    folder.mkdir()
    for name in MODULE._prediction_bundle_is_complete.__globals__.get(
        "REQUIRED_PREDICTION_FILES", (
            "ensemble_prediction.csv",
            "ensemble_prediction_long.csv",
            "ensemble_prediction.npz",
            "ensemble_report.json",
        )
    ):
        (folder / name).write_text("", encoding="utf-8")
    (folder / "ensemble_report.json").write_text(
        json.dumps({"study_instance_uid": "1.2.3"}),
        encoding="utf-8",
    )
    assert MODULE._prediction_bundle_is_complete(folder, "1.2.3")
    assert not MODULE._prediction_bundle_is_complete(folder, "1.2.4")
