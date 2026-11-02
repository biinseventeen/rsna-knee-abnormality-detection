from __future__ import annotations

import json
import zipfile
from pathlib import Path

from rsna_knee.models.dinov2_bundle import (
    find_bundle_root,
    resolve_bundle_root,
    validate_bundle_root,
)
from rsna_knee.models.dinov2_five_role import FIVE_ROLE_ORDER, TARGETS


def _write_fake_bundle(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "architecture": "DINOv2 ViT-S/14 shared 2.5D five-role; target-by-role attention; 12 independent heads",
        "backbone_source": "facebookresearch/dinov2",
        "backbone_model": "dinov2_vits14",
        "backbone_pretrained_weights_included": False,
        "image_size": 224,
        "patch_size": 14,
        "backbone_dim": 384,
        "embed_dim": 256,
        "freeze_early_blocks": 9,
        "n_role_slices": 12,
        "n_role_windows": 10,
        "targets": list(TARGETS),
        "role_order": list(FIVE_ROLE_ORDER),
        "folds": [{"fold": fold} for fold in range(1, 6)],
    }
    (root / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "dinov2_repo" / "dinov2").mkdir(parents=True)
    (root / "dinov2_repo" / "hubconf.py").write_text("# test placeholder\n", encoding="utf-8")
    (root / "dinov2_repo" / "dinov2" / "__init__.py").write_text("# test\n", encoding="utf-8")
    for fold in range(1, 6):
        checkpoint = root / "five_role_2p5d_v2_cv5" / f"fold_{fold}" / "best_checkpoint.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"test checkpoint placeholder")


def test_validate_and_discover_complete_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "some_dataset" / "wrapped_bundle"
    _write_fake_bundle(bundle)

    resolved, manifest = validate_bundle_root(bundle)
    assert resolved == bundle.resolve()
    assert manifest["backbone_model"] == "dinov2_vits14"
    assert find_bundle_root(tmp_path / "some_dataset") == bundle.resolve()


def test_resolve_wrapped_zip_bundle(tmp_path: Path) -> None:
    source_root = tmp_path / "source" / "wrapped_bundle"
    _write_fake_bundle(source_root)
    archive_path = tmp_path / "final_bundle.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(source_root.parent).as_posix())

    destination = tmp_path / "extracted_bundle"
    resolved = resolve_bundle_root(archive_path, destination)
    assert resolved == destination.resolve()
    validate_bundle_root(resolved)
