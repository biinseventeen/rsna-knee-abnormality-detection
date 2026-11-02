"""Resolve and strictly restore the final ViT-B/16 checkpoint artifact."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import torch

from .vit_study import TARGETS, PretrainedViTStudyModel, ViTModelConfig

CHECKPOINT_NAME = "vit_pretrained_best_checkpoint.pt"


class ViTCheckpointError(RuntimeError):
    pass


def _torch_load(source: Any) -> dict[str, Any]:
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch versions
        if hasattr(source, "seek"):
            source.seek(0)
        checkpoint = torch.load(source, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ViTCheckpointError(
            f"Checkpoint root must be dict, got {type(checkpoint).__name__}."
        )
    return checkpoint


def _load_checkpoint_payload(artifact_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load .pt directly or load the one named checkpoint from an artifact ZIP."""
    if artifact_path.suffix.lower() == ".pt":
        checkpoint = _torch_load(artifact_path)
        source_info = {"artifact": str(artifact_path), "member": None}
        return checkpoint, source_info

    if artifact_path.suffix.lower() != ".zip":
        raise ValueError(
            "ViT artifact must be a .pt checkpoint or .zip artifact; "
            f"received: {artifact_path}"
        )

    with zipfile.ZipFile(artifact_path, "r") as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ViTCheckpointError(f"Corrupt ZIP member: {bad_member}")
        matches = [
            name for name in archive.namelist()
            if Path(name).name == CHECKPOINT_NAME and not name.endswith("/")
        ]
        if len(matches) != 1:
            raise ViTCheckpointError(
                f"Expected exactly one {CHECKPOINT_NAME} in {artifact_path}; "
                f"found {len(matches)}: {matches[:5]}"
            )
        member = matches[0]
        with archive.open(member, "r") as stream:
            checkpoint = _torch_load(stream)
    return checkpoint, {"artifact": str(artifact_path), "member": member}


def resolve_vit_artifact(source: str | Path | None, project_root: str | Path) -> Path:
    """Resolve an explicit artifact, otherwise search the project's artifacts/."""
    if source is not None and str(source).strip():
        path = Path(source).expanduser().resolve()
        if path.is_file():
            return path
        if path.is_dir():
            candidates = sorted(
                list(path.rglob(CHECKPOINT_NAME))
                + list(path.rglob("vit_pretrained_centerprior_model_artifacts.zip"))
            )
            candidates = list(dict.fromkeys(p.resolve() for p in candidates))
            if len(candidates) == 1:
                return candidates[0]
            raise FileNotFoundError(
                f"Expected exactly one ViT checkpoint or artifact ZIP under {path}; "
                f"found {len(candidates)}."
            )
        raise FileNotFoundError(f"Configured ViT artifact does not exist: {path}")

    root = Path(project_root).expanduser().resolve()
    artifacts = root / "artifacts"
    if not artifacts.is_dir():
        raise FileNotFoundError(f"Project artifacts directory not found: {artifacts}")

    candidates = sorted(
        list(artifacts.rglob(CHECKPOINT_NAME))
        + list(artifacts.rglob("vit_pretrained_centerprior_model_artifacts.zip"))
    )
    # Prefer the final dated checkpoint artifact path if more than one is present.
    final_candidates = [
        p for p in candidates
        if "vit_pretrained_centerprior_model_artifacts.zip" in p.name
        or (p.name == CHECKPOINT_NAME and "102526" in str(p).replace("\\", "/"))
    ]
    selected = final_candidates or candidates
    selected = list(dict.fromkeys(p.resolve() for p in selected))
    if len(selected) != 1:
        detail = "\n".join(str(p) for p in selected) or "(none)"
        raise FileNotFoundError(
            "Could not uniquely resolve the final ViT artifact. Set VIT_ARTIFACT_SOURCE "
            f"to its exact .pt or .zip path. Candidates:\n{detail}"
        )
    return selected[0]


def load_vit_model(
    artifact: str | Path,
    *,
    device: str | torch.device | None = None,
    runtime_slice_chunk_size: int | None = None,
) -> tuple[PretrainedViTStudyModel, dict[str, Any]]:
    """Build torchvision ViT-B/16 with weights=None and strict-load the trained state."""
    artifact_path = Path(artifact).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)

    checkpoint, source_info = _load_checkpoint_payload(artifact_path)
    checkpoint_targets = list(checkpoint.get("targets", []))
    if checkpoint_targets != list(TARGETS):
        raise ViTCheckpointError(
            "Checkpoint target order mismatch.\n"
            f"Checkpoint={checkpoint_targets}\nExpected={list(TARGETS)}"
        )
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise ViTCheckpointError("Checkpoint has no non-empty model_state.")

    run_config = checkpoint.get("run_config", {})
    if not isinstance(run_config, dict):
        raise ViTCheckpointError("checkpoint.run_config must be a dictionary.")
    config = ViTModelConfig.from_run_config(run_config)
    if runtime_slice_chunk_size is not None:
        config = ViTModelConfig(
            **{
                **config.__dict__,
                "slice_chunk_size": int(runtime_slice_chunk_size),
            }
        )
    config.validate()

    try:
        from torchvision.models import vit_b_16
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Could not import torchvision.models.vit_b_16. Check that your "
            "PyTorch and torchvision versions are compatible."
        ) from exc

    backbone = vit_b_16(weights=None)
    model = PretrainedViTStudyModel(backbone=backbone, config=config)
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ViTCheckpointError(
            "Checkpoint không tương thích với Residual Attention Pooling ViT-B/16. "
            "Hãy chắc chắn đây là vit_pretrained_best_checkpoint.pt của run "
            "Pretrained CenterPrior/Residual Attention, không phải ViT Study-CLS cũ.\n"
            + str(exc)
        ) from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ViTCheckpointError(
            f"Strict load returned incompatible keys: {incompatible}"
        )

    device_obj = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device_obj)
    model.eval()

    info = {
        **source_info,
        "checkpoint_name": CHECKPOINT_NAME,
        "best_epoch": checkpoint.get("best_epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "architecture": run_config.get("architecture", "not recorded"),
        "run_config": {**run_config, "runtime_slice_chunk_size": config.slice_chunk_size},
        "target_order": checkpoint_targets,
        "strict_load": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device_obj),
        "torch_version": torch.__version__,
    }
    # Drop the CPU checkpoint payload; the model now owns the loaded tensors.
    del checkpoint, state
    return model, info
