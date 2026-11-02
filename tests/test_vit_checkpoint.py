import sys
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import zipfile
from pathlib import Path

import torch

from rsna_knee.models.vit_checkpoint import (
    CHECKPOINT_NAME,
    _load_checkpoint_payload,
    resolve_vit_artifact,
)


def test_load_checkpoint_from_zip_member(tmp_path: Path):
    payload = {
        "fold": 1,
        "model_state": {"weight": torch.tensor([1.0, 2.0])},
    }
    checkpoint_path = tmp_path / CHECKPOINT_NAME
    torch.save(payload, checkpoint_path)
    archive_path = tmp_path / "vit_artifacts.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(checkpoint_path, arcname=f"nested/model/{CHECKPOINT_NAME}")

    loaded, source_info = _load_checkpoint_payload(archive_path)
    assert loaded["fold"] == 1
    assert loaded["model_state"]["weight"].tolist() == [1.0, 2.0]
    assert source_info["member"] == f"nested/model/{CHECKPOINT_NAME}"


def test_resolve_explicit_vit_artifact_directory(tmp_path: Path):
    artifact = tmp_path / "vit_pretrained_centerprior_model_artifacts.zip"
    artifact.write_bytes(b"not opened by resolver")
    resolved = resolve_vit_artifact(tmp_path, tmp_path)
    assert resolved == artifact.resolve()
