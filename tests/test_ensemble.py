from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rsna_knee.inference.ensemble import (
    TARGETS,
    EnsembleIntegrationError,
    ModelPrediction,
    blend_probabilities,
    load_dinov2_prediction,
    load_vit_prediction,
    run_local_ensemble,
)

STUDY_UID = "1.2.826.0.1.3680043.8.498.123456789"


def model(name: str, probs: np.ndarray, valid: bool, status: str) -> ModelPrediction:
    return ModelPrediction(
        model_name=name,
        study_uid=STUDY_UID,
        probabilities=np.asarray(probs, dtype=np.float32),
        valid=valid,
        status=status,
        source_npz="test.npz",
        source_report="test.json",
        metadata={},
    )


def test_weighted_blend_both_valid():
    dino = np.linspace(0.1, 0.8, 12, dtype=np.float32)
    vit = np.linspace(0.8, 0.2, 12, dtype=np.float32)
    result = blend_probabilities(
        study_uid=STUDY_UID,
        dinov2=model("DINOv2", dino, True, "ok"),
        vit=model("ViT", vit, True, "success"),
    )
    assert result.blend_status == "weighted_ensemble"
    assert result.effective_dinov2_weight == pytest.approx(0.9)
    assert result.effective_vit_weight == pytest.approx(0.1)
    np.testing.assert_allclose(result.probabilities, 0.9 * dino + 0.1 * vit)


def test_dino_only_fallback():
    dino = np.linspace(0.1, 0.8, 12, dtype=np.float32)
    vit = np.full(12, 0.5, dtype=np.float32)
    result = blend_probabilities(
        study_uid=STUDY_UID,
        dinov2=model("DINOv2", dino, True, "ok"),
        vit=model("ViT", vit, False, "not_usable:no_valid_slices"),
    )
    assert result.blend_status == "dinov2_only"
    np.testing.assert_array_equal(result.probabilities, dino)


def test_vit_only_fallback():
    dino = np.full(12, 0.5, dtype=np.float32)
    vit = np.linspace(0.2, 0.9, 12, dtype=np.float32)
    result = blend_probabilities(
        study_uid=STUDY_UID,
        dinov2=model("DINOv2", dino, False, "all_roles_unusable_fallback_0.5"),
        vit=model("ViT", vit, True, "success"),
    )
    assert result.blend_status == "vit_only"
    np.testing.assert_array_equal(result.probabilities, vit)


def test_neither_valid_uses_explicit_neutral_fallback():
    neutral = np.full(12, 0.5, dtype=np.float32)
    result = blend_probabilities(
        study_uid=STUDY_UID,
        dinov2=model("DINOv2", neutral, False, "all_roles_unusable_fallback_0.5"),
        vit=model("ViT", neutral, False, "not_usable:no_valid_slices"),
    )
    assert result.blend_status == "neutral_fallback_0.5"
    np.testing.assert_array_equal(result.probabilities, neutral)


def test_study_uid_mismatch_rejected():
    dino = model("DINOv2", np.full(12, 0.5), True, "ok")
    vit = ModelPrediction("ViT", "another-study", np.full(12, 0.5), True, "success", "", "", {})
    with pytest.raises(EnsembleIntegrationError, match="StudyInstanceUID mismatch"):
        blend_probabilities(study_uid=STUDY_UID, dinov2=dino, vit=vit)


def test_invalid_probability_rejected():
    dino = model("DINOv2", np.full(12, 1.2), True, "ok")
    vit = model("ViT", np.full(12, 0.5), True, "success")
    with pytest.raises(EnsembleIntegrationError, match=r"within \[0, 1\]"):
        blend_probabilities(study_uid=STUDY_UID, dinov2=dino, vit=vit)


def _write_valid_artifacts(root: Path):
    dino_dir = root / "dino"
    vit_dir = root / "vit"
    dino_dir.mkdir()
    vit_dir.mkdir()
    dino_fold = np.tile(np.linspace(0.1, 0.8, 12, dtype=np.float32), (5, 1))
    dino_mean = dino_fold.mean(axis=0)
    vit_probs = np.linspace(0.8, 0.2, 12, dtype=np.float32)
    np.savez_compressed(
        dino_dir / "dinov2_five_fold_predictions.npz",
        study_uid=np.asarray(STUDY_UID),
        targets=np.asarray(TARGETS),
        status=np.asarray("ok"),
        fold_probabilities=dino_fold,
        ensemble_probabilities=dino_mean,
        fold_std=dino_fold.std(axis=0),
    )
    (dino_dir / "dinov2_five_fold_inference_report.json").write_text(
        json.dumps({
            "study_instance_uid": STUDY_UID,
            "status": "ok",
            "device": "cpu",
            "valid_windows": 50,
            "available_roles": 5,
            "elapsed_seconds": 1.0,
            "fold_info": [{"fold": i, "strict_load": True} for i in range(1, 6)],
        }),
        encoding="utf-8",
    )
    np.savez_compressed(
        vit_dir / "vit_full_study_prediction.npz",
        study_uid=np.asarray(STUDY_UID),
        targets=np.asarray(TARGETS),
        logits=np.zeros(12, dtype=np.float32),
        probabilities=vit_probs,
    )
    (vit_dir / "vit_full_study_inference_report.json").write_text(
        json.dumps({
            "status": "success",
            "study_instance_uid": STUDY_UID,
            "strict_load": True,
            "architecture": "ViT-B/16",
            "input": {"valid_slices": 36, "valid_slices_by_plane": {"Sagittal": 12, "Coronal": 12, "Axial": 12}},
            "output": {"all_finite": True},
            "timing_seconds": {"full_study_inference": 1.0},
        }),
        encoding="utf-8",
    )
    return dino_dir, vit_dir, dino_mean, vit_probs


def test_load_and_save_local_ensemble(tmp_path):
    dino_dir, vit_dir, dino_probs, vit_probs = _write_valid_artifacts(tmp_path)
    dino = load_dinov2_prediction(dino_dir)
    vit = load_vit_prediction(vit_dir)
    assert dino.valid and vit.valid
    result, paths = run_local_ensemble(
        dinov2_output_dir=dino_dir,
        vit_output_dir=vit_dir,
        output_dir=tmp_path / "ensemble",
    )
    np.testing.assert_allclose(result.probabilities, 0.9 * dino_probs + 0.1 * vit_probs)
    assert set(paths) == {"wide_csv", "long_csv", "npz", "report"}
    assert all(path.is_file() for path in paths.values())
    with np.load(paths["npz"], allow_pickle=False) as data:
        np.testing.assert_allclose(data["ensemble_probabilities"], result.probabilities)
        assert data["targets"].astype(str).tolist() == list(TARGETS)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["status"] == "success"
    assert report["blend_status"] == "weighted_ensemble"
