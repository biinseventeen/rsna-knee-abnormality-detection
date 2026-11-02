"""Utilities for validating and loading the final five-fold DINOv2 bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .dinov2_five_role import (
    BACKBONE_DIM,
    DINO_HUB_MODEL,
    EMBED_DIM,
    FIVE_ROLE_ORDER,
    IMAGE_SIZE,
    N_ROLE_SLICES,
    N_ROLE_WINDOWS,
    PATCH_SIZE,
    TARGETS,
    FiveRoleDINOv2_2p5D,
)

MANIFEST_NAME = "bundle_manifest.json"
REPO_DIR_NAME = "dinov2_repo"
CHECKPOINT_RELATIVE_PATH = Path("five_role_2p5d_v2_cv5")


class DinoBundleError(RuntimeError):
    """Raised when the supplied artifact is not the expected final DINOv2 bundle."""


def _required_checkpoint_paths(root: Path) -> list[Path]:
    return [
        root / CHECKPOINT_RELATIVE_PATH / f"fold_{fold}" / "best_checkpoint.pt"
        for fold in range(1, 6)
    ]


def validate_bundle_root(bundle_root: str | Path) -> tuple[Path, dict[str, Any]]:
    """Validate an extracted ``five_role_v2_cv5`` bundle and read its manifest.

    This deliberately rejects a small/older ``dinov2-artifact.zip`` that does
    not contain all five final fold checkpoints and the local DINOv2 source.
    """
    root = Path(bundle_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DINOv2 bundle directory not found: {root}")

    manifest_path = root / MANIFEST_NAME
    repo_dir = root / REPO_DIR_NAME
    required = [manifest_path, repo_dir / "hubconf.py", repo_dir / "dinov2"]
    required.extend(_required_checkpoint_paths(root))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise DinoBundleError(
            "This directory is not a complete final five-fold DINOv2 bundle. "
            "Missing required items:\n- " + "\n- ".join(missing)
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_bundle_manifest(manifest)
    return root, manifest


def validate_bundle_manifest(manifest: dict[str, Any]) -> None:
    """Check the bundle manifest against the model contract used by the notebook."""
    expected = {
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "backbone_dim": BACKBONE_DIM,
        "embed_dim": EMBED_DIM,
        "n_role_slices": N_ROLE_SLICES,
        "n_role_windows": N_ROLE_WINDOWS,
    }
    errors: list[str] = []
    for key, value in expected.items():
        actual = manifest.get(key)
        if actual != value:
            errors.append(f"{key}: expected {value!r}, found {actual!r}")

    if list(manifest.get("targets", [])) != list(TARGETS):
        errors.append("targets order does not match the final Kaggle target order")
    if list(manifest.get("role_order", [])) != list(FIVE_ROLE_ORDER):
        errors.append("role_order does not match the final five-role order")
    if manifest.get("backbone_model") != DINO_HUB_MODEL:
        errors.append(
            f"backbone_model: expected {DINO_HUB_MODEL!r}, "
            f"found {manifest.get('backbone_model')!r}"
        )
    if manifest.get("backbone_pretrained_weights_included") is not False:
        errors.append(
            "expected backbone_pretrained_weights_included=false; the fold "
            "checkpoints themselves must contain the full fine-tuned backbone"
        )

    if errors:
        raise DinoBundleError("DINOv2 bundle manifest mismatch:\n- " + "\n- ".join(errors))


def find_bundle_root(search_root: str | Path) -> Path:
    """Find exactly one complete extracted bundle below ``search_root``."""
    search_root = Path(search_root).expanduser().resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"Artifact search directory does not exist: {search_root}")

    candidates: list[Path] = []
    for manifest_path in search_root.rglob(MANIFEST_NAME):
        candidate = manifest_path.parent
        try:
            validate_bundle_root(candidate)
        except (DinoBundleError, FileNotFoundError, json.JSONDecodeError):
            continue
        candidates.append(candidate.resolve())

    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        details = "\n".join(str(p) for p in candidates) or "(none)"
        raise DinoBundleError(
            f"Expected exactly one complete final DINOv2 bundle under {search_root}; "
            f"found {len(candidates)}:\n{details}\n"
            "Set DINO_BUNDLE_ROOT explicitly in the notebook if the bundle is stored elsewhere."
        )
    return candidates[0]


def _torch_load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load a trusted project checkpoint across PyTorch versions.

    The competition-generated checkpoint is a trusted local artifact. Explicit
    ``weights_only=False`` is required for PyTorch versions whose default changed.
    Do not use this helper for untrusted files from unknown sources.
    """
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # PyTorch versions before the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise DinoBundleError(
            f"Checkpoint root must be a dict, received {type(checkpoint).__name__}."
        )
    return checkpoint


def _validate_checkpoint_metadata(
    checkpoint: dict[str, Any], fold: int
) -> None:
    actual_fold = int(checkpoint.get("fold", -1))
    if actual_fold != fold:
        raise DinoBundleError(
            f"Fold metadata mismatch: expected fold {fold}, found {actual_fold}."
        )
    if list(checkpoint.get("targets", [])) != list(TARGETS):
        raise DinoBundleError(f"Fold {fold}: target order mismatch.")
    if list(checkpoint.get("role_order", [])) != list(FIVE_ROLE_ORDER):
        raise DinoBundleError(f"Fold {fold}: role order mismatch.")
    state = checkpoint.get("model_state")
    if not isinstance(state, dict) or not state:
        raise DinoBundleError(f"Fold {fold}: model_state missing or empty.")


def build_local_backbone(repo_dir: str | Path) -> torch.nn.Module:
    """Instantiate DINOv2 ViT-S/14 from the bundle's local source snapshot.

    ``pretrained=False`` is intentional: no network access or Hub download is
    required. The complete fine-tuned backbone weights are loaded from the fold
    checkpoint immediately afterwards.
    """
    repo_dir = Path(repo_dir).expanduser().resolve()
    if not (repo_dir / "hubconf.py").is_file():
        raise FileNotFoundError(f"Local DINOv2 hubconf.py not found: {repo_dir}")
    if not (repo_dir / "dinov2").is_dir():
        raise FileNotFoundError(f"Local DINOv2 source package not found: {repo_dir / 'dinov2'}")

    try:
        backbone = torch.hub.load(
            str(repo_dir), DINO_HUB_MODEL, source="local", pretrained=False
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not instantiate DINOv2 from the bundle's local source. "
            "Check PyTorch/torchvision compatibility and verify dinov2_repo/. "
            "No network download is attempted by this loader."
        ) from exc

    if not hasattr(backbone, "forward_features"):
        raise DinoBundleError("Local backbone has no forward_features() method.")
    if not hasattr(backbone, "blocks") or len(backbone.blocks) != 12:
        raise DinoBundleError("Local backbone is not the expected DINOv2 ViT-S/14 (12 blocks).")
    if not hasattr(backbone, "norm"):
        raise DinoBundleError("Local DINOv2 backbone has no final LayerNorm named 'norm'.")
    return backbone


def load_dinov2_fold(
    bundle_root: str | Path,
    fold: int = 1,
    device: str | torch.device | None = None,
    encoder_chunk_size: int | None = None,
) -> tuple[FiveRoleDINOv2_2p5D, dict[str, Any]]:
    """Build the exact model and load one fold with ``strict=True``.

    Only one fold is loaded. This is intentional for the local demo so that all
    five large model copies are not held in RAM at once.
    """
    if fold not in (1, 2, 3, 4, 5):
        raise ValueError("fold must be one of 1, 2, 3, 4, 5")

    root, manifest = validate_bundle_root(bundle_root)
    device_obj = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if encoder_chunk_size is None:
        encoder_chunk_size = 8 if device_obj.type == "cuda" else 2

    checkpoint_path = (
        root / CHECKPOINT_RELATIVE_PATH / f"fold_{fold}" / "best_checkpoint.pt"
    )
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    _validate_checkpoint_metadata(checkpoint, fold)

    backbone = build_local_backbone(root / REPO_DIR_NAME)
    model = FiveRoleDINOv2_2p5D(
        backbone=backbone,
        image_size=int(manifest["image_size"]),
        backbone_dim=int(manifest["backbone_dim"]),
        embed_dim=int(manifest["embed_dim"]),
        freeze_early_blocks=int(manifest.get("freeze_early_blocks", 9)),
        encoder_chunk_size=int(encoder_chunk_size),
    )

    # The central architecture-compatibility check: no missing or extra keys.
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        # Strict loading normally raises before this point. Keep this guard as
        # an explicit invariant in case PyTorch's return contract changes.
        raise DinoBundleError(
            "strict=True returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    model.to(device_obj)
    model.eval()

    info = {
        "fold": int(checkpoint.get("fold", fold)),
        "best_epoch": int(checkpoint.get("best_epoch", -1)),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "signature": str(checkpoint.get("signature", "")),
        "device": str(device_obj),
        "encoder_chunk_size": int(encoder_chunk_size),
        "strict_load": True,
        "checkpoint_path": str(checkpoint_path),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "manifest_architecture": str(manifest.get("architecture", "")),
        "targets": list(TARGETS),
        "role_order": list(FIVE_ROLE_ORDER),
    }

    del checkpoint
    return model, info


def resolve_bundle_root(
    bundle_source: str | Path,
    extraction_dir: str | Path | None = None,
) -> Path:
    """Resolve an extracted bundle directory or safely extract a final bundle ZIP.

    ZIPs may contain the bundle files at archive root or beneath one enclosing
    directory. Only the subtree containing the single bundle manifest is
    extracted. The five fold checkpoints are included; no checkpoint is omitted.
    """
    source = Path(bundle_source).expanduser().resolve()
    if source.is_dir():
        try:
            validate_bundle_root(source)
            return source
        except DinoBundleError:
            return find_bundle_root(source)

    if not source.is_file() or source.suffix.lower() != ".zip":
        raise FileNotFoundError(
            "DINO_BUNDLE_SOURCE must be an extracted bundle directory or the "
            f"final five-fold bundle ZIP. Received: {source}"
        )

    import shutil
    import zipfile
    from pathlib import PurePosixPath

    with zipfile.ZipFile(source, "r") as archive:
        names = archive.namelist()
        manifest_members = [
            name.replace("\\", "/")
            for name in names
            if name.replace("\\", "/").endswith("/" + MANIFEST_NAME)
            or name.replace("\\", "/") == MANIFEST_NAME
        ]
        if len(manifest_members) != 1:
            raise DinoBundleError(
                "Expected exactly one bundle_manifest.json in DINOv2 ZIP; "
                f"found {len(manifest_members)}."
            )

        manifest_member = manifest_members[0]
        prefix = manifest_member.rsplit("/", 1)[0] + "/" if "/" in manifest_member else ""
        for name in names:
            normalized = name.replace("\\", "/")
            pure = PurePosixPath(normalized)
            if pure.is_absolute() or ".." in pure.parts:
                raise DinoBundleError(f"Unsafe path in DINOv2 ZIP: {name}")

        destination = (
            Path(extraction_dir).expanduser().resolve()
            if extraction_dir is not None
            else source.with_suffix("").with_name(source.stem + "_extracted")
        )
        if destination.exists():
            try:
                validate_bundle_root(destination)
                return destination
            except (DinoBundleError, FileNotFoundError, json.JSONDecodeError):
                shutil.rmtree(destination, ignore_errors=True)

        temporary = destination.with_name(destination.name + ".extracting")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=True)
        try:
            for name in names:
                normalized = name.replace("\\", "/")
                if prefix and not normalized.startswith(prefix):
                    continue
                relative = normalized[len(prefix):] if prefix else normalized
                if not relative or relative.endswith("/"):
                    continue
                relative_path = PurePosixPath(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise DinoBundleError(f"Unsafe bundle-relative path: {name}")
                target = temporary.joinpath(*relative_path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            validate_bundle_root(temporary)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(destination, ignore_errors=True)
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    validate_bundle_root(destination)
    return destination
