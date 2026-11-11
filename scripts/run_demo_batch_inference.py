"""
Batch inference for multiple local RSNA demo studies.

Runs the already-validated Phase 2A/2C/2D/2E and Phase 3 components for each
exported study package, then writes Phase 3 bundles under:
    demo/output/ensemble/<StudyInstanceUID>/

This makes the outputs discoverable by the Phase 4 multi-study dashboard.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


FIRST_SAMPLE_UID = (
    "1.2.826.0.1.3680043.8.498."
    "10095687747295410396510538520594649149"
)


def find_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).expanduser().resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if (
            (candidate / "src" / "rsna_knee" / "dicom" / "sample.py").is_file()
            and (candidate / "src" / "rsna_knee" / "inference").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy project root. Hãy chạy notebook từ project có "
        "src/rsna_knee/dicom/sample.py."
    )


def discover_study_packages(studies_root: str | Path) -> list[Path]:
    """Find one exported study package at each manifest.json location."""
    root = Path(studies_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục studies: {root}")

    roots: list[Path] = []
    for manifest_path in root.rglob("manifest.json"):
        package_root = manifest_path.parent
        if (
            (package_root / "series_metadata.csv").is_file()
            and (package_root / "series_selection.csv").is_file()
            and (package_root / "dicom").is_dir()
        ):
            roots.append(package_root.resolve())

    # Keep deterministic order and remove duplicate paths.
    return sorted(set(roots), key=lambda p: str(p).casefold())


def _read_manifest_uid(package_root: Path) -> str:
    manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest.get("source", {}).get("study_instance_uid", "")).strip()


def _study_uid_from_package(package_root: Path) -> str:
    dicom_root = package_root / "dicom"
    study_dirs = [p for p in dicom_root.iterdir() if p.is_dir()]
    if len(study_dirs) != 1:
        raise ValueError(
            f"{package_root}: expected exactly one study folder under dicom/, "
            f"found {len(study_dirs)}."
        )
    uid_from_dicom = study_dirs[0].name
    uid_from_manifest = _read_manifest_uid(package_root)
    if uid_from_manifest and uid_from_manifest != uid_from_dicom:
        raise ValueError(
            f"{package_root}: manifest UID {uid_from_manifest} differs from "
            f"DICOM folder UID {uid_from_dicom}."
        )
    return uid_from_dicom


def _save_preprocessing_caches(study, dino_pre, vit_pre, output_root: Path) -> tuple[Path, Path]:
    dino_cache_dir = output_root / "dinov2_preprocessing"
    vit_cache_dir = output_root / "vit_preprocessing"
    dino_cache_dir.mkdir(parents=True, exist_ok=True)
    vit_cache_dir.mkdir(parents=True, exist_ok=True)

    dino_cache = dino_cache_dir / "dinov2_preprocessed_sample.npz"
    np.savez_compressed(
        dino_cache,
        study_uid=np.asarray(study.study_uid),
        images=dino_pre.images,
        window_valid=dino_pre.window_valid,
        window_positions=dino_pre.window_positions,
        role_series_uids=np.asarray(
            [dino_pre.role_series_uids[k] for k in dino_pre.role_series_uids]
        ),
    )
    (dino_cache_dir / "dinov2_preprocessing_summary.json").write_text(
        json.dumps(
            {
                "study_instance_uid": study.study_uid,
                "valid_windows": dino_pre.n_valid_windows,
                "available_roles": dino_pre.n_available_roles,
                "role_series_uids": dino_pre.role_series_uids,
                "diagnostics": dino_pre.diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    vit_cache = vit_cache_dir / "vit_preprocessed_sample.npz"
    np.savez_compressed(
        vit_cache,
        study_uid=np.asarray(study.study_uid),
        images=vit_pre.images,
        valid=vit_pre.valid,
        plane_ids=vit_pre.plane_ids,
        fluid=vit_pre.fluid,
        fat=vit_pre.fat,
        positions=vit_pre.positions,
    )
    (vit_cache_dir / "vit_preprocessing_summary.json").write_text(
        json.dumps(
            {
                "study_instance_uid": study.study_uid,
                "valid_slices": vit_pre.n_valid_slices,
                "series_used": vit_pre.n_series_used,
                "series_uids": vit_pre.series_uids,
                "diagnostics": vit_pre.diagnostics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return dino_cache, vit_cache


def _prediction_bundle_is_complete(directory: Path, expected_uid: str) -> bool:
    required = (
        "ensemble_prediction.csv",
        "ensemble_prediction_long.csv",
        "ensemble_prediction.npz",
        "ensemble_report.json",
    )
    if not all((directory / name).is_file() for name in required):
        return False
    try:
        report = json.loads((directory / "ensemble_report.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(report.get("study_instance_uid", "")) == expected_uid


def run_batch_inference(
    *,
    studies_root: str | Path,
    project_root: str | Path | None = None,
    dino_bundle_source: str | Path | None = None,
    vit_artifact_source: str | Path | None = None,
    max_studies: int = 4,
    exclude_uids: tuple[str, ...] = (FIRST_SAMPLE_UID,),
    device: str | None = None,
    dino_encoder_chunk_size: int | None = None,
    vit_slice_chunk_size: int | None = None,
    skip_complete: bool = True,
) -> list[dict[str, Any]]:
    """Run preprocessing, both model inferences and Phase 3 for each new study."""
    project = find_project_root(project_root)
    src_path = project / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # Imports are delayed until project root is discovered and src is on sys.path.
    from rsna_knee.dicom.sample import load_demo_sample
    from rsna_knee.preprocessing.dinov2 import preprocess_dinov2_study
    from rsna_knee.preprocessing.vit import preprocess_vit_study
    from rsna_knee.models.dinov2_bundle import find_bundle_root, resolve_bundle_root
    from rsna_knee.models.vit_checkpoint import resolve_vit_artifact, load_vit_model
    from rsna_knee.inference.dinov2_ensemble import (
        run_dinov2_five_fold_inference,
        save_dinov2_inference_result,
    )
    from rsna_knee.inference.vit_full_study import (
        predict_vit_full_study,
        save_vit_full_study_outputs,
    )
    from rsna_knee.inference.ensemble import run_local_ensemble

    device_obj = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng PyTorch không thấy CUDA.")

    if dino_encoder_chunk_size is None:
        dino_encoder_chunk_size = 8 if device_obj.type == "cuda" else 1
    if vit_slice_chunk_size is None:
        vit_slice_chunk_size = 8 if device_obj.type == "cuda" else 2

    if dino_bundle_source:
        dino_bundle_root = resolve_bundle_root(dino_bundle_source)
    else:
        dino_bundle_root = find_bundle_root(project / "artifacts")

    vit_artifact = resolve_vit_artifact(vit_artifact_source, project)

    package_roots = discover_study_packages(studies_root)
    excluded = {str(uid).strip() for uid in exclude_uids if str(uid).strip()}
    eligible: list[tuple[Path, str]] = []
    rejected: list[dict[str, str]] = []

    seen_uids: dict[str, Path] = {}
    for package_root in package_roots:
        try:
            uid = _study_uid_from_package(package_root)
            if uid in excluded:
                rejected.append({"path": str(package_root), "uid": uid, "reason": "excluded_first_sample"})
                continue
            if uid in seen_uids:
                raise ValueError(
                    f"Duplicate StudyInstanceUID {uid} in {seen_uids[uid]} and {package_root}."
                )
            seen_uids[uid] = package_root
            eligible.append((package_root, uid))
        except Exception as exc:
            rejected.append(
                {"path": str(package_root), "uid": "", "reason": f"{type(exc).__name__}: {exc}"}
            )

    if not eligible:
        raise RuntimeError(
            f"Không tìm thấy study hợp lệ dưới {studies_root}. "
            "Mỗi package cần manifest.json, series_metadata.csv, "
            "series_selection.csv và dicom/."
        )
    if max_studies is not None:
        eligible = eligible[: int(max_studies)]
    if len(eligible) != int(max_studies):
        raise RuntimeError(
            f"Yêu cầu chạy {max_studies} studies nhưng chỉ tìm thấy {len(eligible)} "
            f"package hợp lệ. Hãy kiểm tra {studies_root}."
        )

    output_base = project / "demo" / "output"
    dino_base = output_base / "dinov2_full_inference_by_study"
    vit_base = output_base / "vit_full_inference_by_study"
    ensemble_base = output_base / "ensemble"
    run_log_path = output_base / "multi_study_inference_log.csv"
    for path in (dino_base, vit_base, ensemble_base):
        path.mkdir(parents=True, exist_ok=True)

    print(f"Project: {project}")
    print(f"Studies root: {Path(studies_root).resolve()}")
    print(f"Device: {device_obj}")
    print(f"DINOv2 bundle: {dino_bundle_root}")
    print(f"ViT artifact: {vit_artifact}")
    print(f"Eligible new studies: {len(eligible)}")
    if rejected:
        print(f"Skipped package paths: {len(rejected)}")

    # Strictly restore the ViT model once and reuse it across studies.
    vit_restore_started = time.perf_counter()
    vit_model, vit_model_info = load_vit_model(
        vit_artifact,
        device=device_obj,
        runtime_slice_chunk_size=vit_slice_chunk_size,
    )
    vit_restore_seconds = time.perf_counter() - vit_restore_started

    rows: list[dict[str, Any]] = []
    for index, (package_root, expected_uid) in enumerate(eligible, start=1):
        row: dict[str, Any] = {
            "study_index": index,
            "study_uid": expected_uid,
            "package_root": str(package_root),
            "status": "started",
            "message": "",
        }
        ensemble_dir = ensemble_base / expected_uid
        if skip_complete and _prediction_bundle_is_complete(ensemble_dir, expected_uid):
            row["status"] = "skipped_already_complete"
            row["message"] = "A valid Phase 3 bundle already exists for this Study UID."
            rows.append(row)
            print(f"[{index}/{len(eligible)}] SKIP {expected_uid}: complete bundle exists.")
            continue

        try:
            print(f"\n[{index}/{len(eligible)}] Loading {expected_uid} from {package_root}")
            study = load_demo_sample(package_root)
            if study.study_uid != expected_uid:
                raise ValueError(
                    f"Loaded UID {study.study_uid} differs from expected UID {expected_uid}."
                )

            report_path = study.root / "radiology_report.txt"
            row["has_radiology_report"] = bool(report_path.is_file())
            if not report_path.is_file():
                print("WARNING: radiology_report.txt is missing; prediction can still run.")

            # Phase 2A / 2D: preserve the official model-specific preprocessing.
            dino_pre = preprocess_dinov2_study(study)
            vit_pre = preprocess_vit_study(study)
            row["dino_valid_windows"] = dino_pre.n_valid_windows
            row["dino_available_roles"] = dino_pre.n_available_roles
            row["vit_valid_slices"] = vit_pre.n_valid_slices

            study_output_base = output_base / "by_study" / expected_uid
            dino_cache, vit_cache = _save_preprocessing_caches(
                study, dino_pre, vit_pre, study_output_base
            )

            # Phase 2C: final five-fold DINOv2 probabilities.
            dino_output_dir = dino_base / expected_uid
            dino_result = run_dinov2_five_fold_inference(
                images=dino_pre.images,
                window_valid=dino_pre.window_valid,
                window_positions=dino_pre.window_positions,
                bundle_root=dino_bundle_root,
                study_uid=expected_uid,
                device=device_obj,
                encoder_chunk_size=dino_encoder_chunk_size,
                use_amp=(device_obj.type == "cuda"),
                verbose=True,
            )
            save_dinov2_inference_result(dino_result, dino_output_dir)
            row["dino_status"] = dino_result.status

            # Phase 2E: final full-study ViT-B/16 probabilities.
            vit_payload = {
                "images": vit_pre.images,
                "valid": vit_pre.valid,
                "plane_ids": vit_pre.plane_ids,
                "fluid": vit_pre.fluid,
                "fat": vit_pre.fat,
                "positions": vit_pre.positions,
            }
            vit_started = time.perf_counter()
            vit_prediction = predict_vit_full_study(
                vit_model,
                vit_payload,
                study_uid=expected_uid,
                device=device_obj,
            )
            vit_elapsed = time.perf_counter() - vit_started
            vit_output_dir = vit_base / expected_uid
            save_vit_full_study_outputs(
                vit_prediction,
                vit_output_dir,
                model_info=vit_model_info,
                restore_seconds=vit_restore_seconds,
                inference_seconds=vit_elapsed,
                cache_path=vit_cache,
            )
            row["vit_status"] = "success"

            # Phase 3: exact 90/10 weighted blend and per-study provenance.
            prediction, output_paths = run_local_ensemble(
                dinov2_output_dir=dino_output_dir,
                vit_output_dir=vit_output_dir,
                output_dir=ensemble_dir,
                dinov2_weight=0.90,
            )
            row["ensemble_status"] = prediction.blend_status
            row["status"] = "success"
            row["message"] = f"Saved {len(output_paths)} Phase 3 files."
            print(
                f"[{index}/{len(eligible)}] PASS {expected_uid} | "
                f"blend={prediction.blend_status} | output={ensemble_dir}",
                flush=True,
            )

        except Exception as exc:
            row["status"] = "failed"
            row["message"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            print(f"[{index}/{len(eligible)}] FAILED {expected_uid}: {row['message']}", flush=True)

        rows.append(row)
        # Persist progress after each study so interrupted runs can resume.
        output_base.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for item in rows for key in item.keys() if key != "traceback"})
        with run_log_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([{k: v for k, v in item.items() if k != "traceback"} for item in rows])

        gc_collector = getattr(__import__("gc"), "collect")
        gc_collector()
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    # Release the shared ViT after all studies.
    del vit_model
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()

    success_count = sum(row["status"] in {"success", "skipped_already_complete"} for row in rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    print("\nBATCH INFERENCE SUMMARY")
    print(f"Completed or already complete: {success_count}/{len(eligible)}")
    print(f"Failed: {failed_count}/{len(eligible)}")
    print(f"Log: {run_log_path}")
    print(f"Phase 3 bundles: {ensemble_base}")
    if failed_count:
        print("Some studies failed. See multi_study_inference_log.csv for per-study details.")
    else:
        print("ALL SELECTED STUDIES HAVE A PHASE 3 BUNDLE.")
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 + Phase 3 for multiple RSNA studies.")
    parser.add_argument("--studies-root", default=r"C:\rsna_demo\studies")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--dino-bundle", default=None)
    parser.add_argument("--vit-artifact", default=None)
    parser.add_argument("--max-studies", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dino-chunk", type=int, default=None)
    parser.add_argument("--vit-chunk", type=int, default=None)
    parser.add_argument("--no-skip-complete", action="store_true")
    args = parser.parse_args()

    run_batch_inference(
        studies_root=args.studies_root,
        project_root=args.project_root,
        dino_bundle_source=args.dino_bundle,
        vit_artifact_source=args.vit_artifact,
        max_studies=args.max_studies,
        device=args.device,
        dino_encoder_chunk_size=args.dino_chunk,
        vit_slice_chunk_size=args.vit_chunk,
        skip_complete=not args.no_skip_complete,
    )


if __name__ == "__main__":
    main()
