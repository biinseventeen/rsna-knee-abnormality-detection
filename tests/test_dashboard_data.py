import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsna_knee.ui.dashboard_data import (  # noqa: E402
    DashboardDataError,
    TARGETS,
    load_prediction_bundle,
)


def _write_bundle(
    root: Path,
    *,
    status: str = "weighted_ensemble",
    report_uid: str = "study-001",
    reverse_targets: bool = False,
):
    root.mkdir(parents=True, exist_ok=True)
    dino = np.linspace(0.1, 0.8, 12, dtype=np.float32)
    vit = np.linspace(0.2, 0.9, 12, dtype=np.float32)
    dino_valid, vit_valid = {
        "weighted_ensemble": (True, True),
        "dinov2_only": (True, False),
        "vit_only": (False, True),
        "neutral_fallback_0.5": (False, False),
    }[status]
    effective = {
        "weighted_ensemble": (0.9, 0.1),
        "dinov2_only": (1.0, 0.0),
        "vit_only": (0.0, 1.0),
        "neutral_fallback_0.5": (0.0, 0.0),
    }[status]
    if status == "weighted_ensemble":
        ensemble = 0.9 * dino + 0.1 * vit
    elif status == "dinov2_only":
        ensemble = dino.copy()
    elif status == "vit_only":
        ensemble = vit.copy()
    else:
        ensemble = np.full(12, 0.5, dtype=np.float32)

    wide = {"StudyInstanceUID": ["study-001"]}
    wide.update({target: [float(ensemble[i])] for i, target in enumerate(TARGETS)})
    pd.DataFrame(wide).to_csv(root / "ensemble_prediction.csv", index=False)

    order = list(range(12))[::-1] if reverse_targets else list(range(12))
    target_order = list(TARGETS[::-1] if reverse_targets else TARGETS)
    long = pd.DataFrame(
        {
            "StudyInstanceUID": ["study-001"] * 12,
            "target": target_order,
            "dinov2_probability": dino[order],
            "vit_probability": vit[order],
            "dinov2_valid": [dino_valid] * 12,
            "vit_valid": [vit_valid] * 12,
            "effective_dinov2_weight": [effective[0]] * 12,
            "effective_vit_weight": [effective[1]] * 12,
            "ensemble_probability": ensemble[order],
            "blend_status": [status] * 12,
        }
    )
    long.to_csv(root / "ensemble_prediction_long.csv", index=False)

    np.savez_compressed(
        root / "ensemble_prediction.npz",
        study_uid=np.asarray("study-001"),
        targets=np.asarray(TARGETS),
        dinov2_probabilities=dino,
        vit_probabilities=vit,
        ensemble_probabilities=ensemble,
        dinov2_valid=np.asarray(dino_valid),
        vit_valid=np.asarray(vit_valid),
        configured_weights=np.asarray([0.9, 0.1], dtype=np.float32),
        effective_weights=np.asarray(effective, dtype=np.float32),
        blend_status=np.asarray(status),
    )

    report = {
        "status": "fallback" if status == "neutral_fallback_0.5" else "success",
        "blend_status": status,
        "study_instance_uid": report_uid,
        "targets": list(TARGETS),
        "models": {
            "dinov2": {"valid": dino_valid, "status": "ok", "metadata": {"fold_count": 5}},
            "vit_b16": {"valid": vit_valid, "status": "success", "metadata": {"valid_slices": 36}},
        },
        "blend": {
            "configured_weights": {"dinov2": 0.9, "vit_b16": 0.1},
            "effective_weights": {"dinov2": effective[0], "vit_b16": effective[1]},
        },
    }
    (root / "ensemble_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_load_prediction_bundle_validates_all_four_phase3_artifacts(tmp_path):
    _write_bundle(tmp_path)
    bundle = load_prediction_bundle(tmp_path)
    assert bundle.study_uid == "study-001"
    assert bundle.targets == TARGETS
    assert bundle.ensemble.shape == (12,)
    assert np.allclose(bundle.ensemble, 0.9 * bundle.dinov2 + 0.1 * bundle.vit)


def test_rejects_study_uid_mismatch(tmp_path):
    _write_bundle(tmp_path, report_uid="another-study")
    with pytest.raises(DashboardDataError, match="StudyInstanceUID"):
        load_prediction_bundle(tmp_path)


def test_rejects_noncanonical_target_order(tmp_path):
    _write_bundle(tmp_path, reverse_targets=True)
    with pytest.raises(DashboardDataError, match="canonical 12-target"):
        load_prediction_bundle(tmp_path)


def test_rejects_invalid_probability_range(tmp_path):
    _write_bundle(tmp_path)
    path = tmp_path / "ensemble_prediction_long.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "ensemble_probability"] = 1.5
    frame.to_csv(path, index=False)
    with pytest.raises(DashboardDataError, match=r"nằm ngoài \[0, 1\]"):
        load_prediction_bundle(tmp_path)


def test_rejects_csv_formula_mismatch(tmp_path):
    _write_bundle(tmp_path)
    path = tmp_path / "ensemble_prediction_long.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "ensemble_probability"] += 0.05
    frame.to_csv(path, index=False)
    with pytest.raises(DashboardDataError, match="Prediction values không khớp"):
        load_prediction_bundle(tmp_path)


def test_rejects_npz_probability_mismatch(tmp_path):
    _write_bundle(tmp_path)
    path = tmp_path / "ensemble_prediction.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["dinov2_probabilities"][0] = 0.99
    np.savez_compressed(path, **arrays)
    with pytest.raises(DashboardDataError, match="DINOv2 CSV/NPZ"):
        load_prediction_bundle(tmp_path)


@pytest.mark.parametrize(
    "status",
    ["dinov2_only", "vit_only", "neutral_fallback_0.5"],
)
def test_validates_each_phase3_fallback_policy(tmp_path, status):
    _write_bundle(tmp_path, status=status)
    bundle = load_prediction_bundle(tmp_path)
    if status == "dinov2_only":
        assert np.allclose(bundle.ensemble, bundle.dinov2)
    elif status == "vit_only":
        assert np.allclose(bundle.ensemble, bundle.vit)
    else:
        assert np.allclose(bundle.ensemble, 0.5)
