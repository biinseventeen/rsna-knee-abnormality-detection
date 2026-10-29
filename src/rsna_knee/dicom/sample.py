"""Load and audit a compact RSNA knee DICOM study exported from Kaggle.

This module deliberately stops before model-specific image preprocessing. It
validates the local sample package, verifies DICOM identities, and orders slices
using the same ordering priority as the final Kaggle ensemble notebook.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import numpy as np
import pandas as pd

REQUIRED_METADATA_COLUMNS = {
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Anatomical_Plane",
    "Fluid_Sensitive",
    "Fat_Suppression",
}
REQUIRED_SELECTION_COLUMNS = {"model", "role_or_plane", "SeriesInstanceUID"}
HEADER_TAGS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "InstanceNumber",
    "Rows",
    "Columns",
    "Modality",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PhotometricInterpretation",
    "NumberOfFrames",
]


@dataclass(frozen=True)
class SliceHeader:
    """Header-only information for one DICOM file."""

    path: Path
    filename: str
    study_uid: str = ""
    series_uid: str = ""
    sop_uid: str = ""
    instance_number: float | None = None
    projected_position_mm: float | None = None
    rows: int | None = None
    columns: int | None = None
    modality: str = ""
    header_error: str = ""
    sort_index: int = -1


@dataclass
class DicomSeries:
    """One selected DICOM series, including its metadata and ordered files."""

    series_uid: str
    directory: Path
    metadata: dict[str, Any]
    files: list[SliceHeader]
    sort_mode: str
    selected_for: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def usable_header_count(self) -> int:
        return sum(not item.header_error for item in self.files)


@dataclass
class DemoStudy:
    """All local DICOM and metadata for one exported study."""

    root: Path
    study_uid: str
    manifest: dict[str, Any]
    metadata: pd.DataFrame
    selection: pd.DataFrame
    series: dict[str, DicomSeries]
    extracted_from: Path | None = None

    def series_for_dino_role(self, role: str) -> DicomSeries | None:
        rows = self.selection[
            self.selection["model"].astype(str).str.casefold().eq("dinov2")
            & self.selection["role_or_plane"].astype(str).eq(role)
        ]
        if rows.empty:
            return None
        uid = str(rows.iloc[0]["SeriesInstanceUID"]).strip()
        return self.series.get(uid) if uid else None

    def series_for_vit_plane(self, plane: str) -> DicomSeries | None:
        rows = self.selection[
            self.selection["model"].astype(str).str.casefold().eq("vit-b/16")
            & self.selection["role_or_plane"].astype(str).eq(plane)
        ]
        if rows.empty:
            return None
        uid = str(rows.iloc[0]["SeriesInstanceUID"]).strip()
        return self.series.get(uid) if uid else None

    def series_summary(self) -> pd.DataFrame:
        rows = []
        for uid, item in self.series.items():
            rows.append(
                {
                    "SeriesInstanceUID": uid,
                    "plane": item.metadata.get("Anatomical_Plane", ""),
                    "fluid_sensitive": item.metadata.get("Fluid_Sensitive", ""),
                    "fat_suppression": item.metadata.get("Fat_Suppression", ""),
                    "n_dicom_files": item.n_files,
                    "valid_headers": item.usable_header_count,
                    "sort_mode": item.sort_mode,
                    "selected_for": ";".join(item.selected_for),
                    "warnings": "; ".join(item.warnings),
                }
            )
        return pd.DataFrame(rows)


def _natural_key(name: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _find_sample_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "manifest.json").is_file() and (path / "series_metadata.csv").is_file():
        return path
    nested = path / "demo_sample"
    if (nested / "manifest.json").is_file() and (nested / "series_metadata.csv").is_file():
        return nested

    matches = [
        candidate
        for candidate in path.rglob("manifest.json")
        if (candidate.parent / "series_metadata.csv").is_file()
    ]
    if len(matches) == 1:
        return matches[0].parent
    if not matches:
        raise FileNotFoundError(
            f"Không tìm thấy manifest.json + series_metadata.csv trong: {path}"
        )
    raise RuntimeError(
        "Tìm thấy nhiều demo_sample package. Hãy truyền trực tiếp thư mục chứa manifest.json."
    )


def _safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    max_uncompressed_bytes: int = 4 * 1024**3,
    max_members: int = 100_000,
) -> Path:
    """Extract an export ZIP while rejecting path traversal and oversized archives."""
    archive_path = archive_path.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) > max_members:
            raise ValueError(f"ZIP có quá nhiều file ({len(members)} > {max_members}).")
        uncompressed = sum(info.file_size for info in members)
        if uncompressed > max_uncompressed_bytes:
            raise ValueError(
                f"ZIP sau giải nén là {uncompressed / 1024**3:.2f} GiB, vượt giới hạn "
                f"{max_uncompressed_bytes / 1024**3:.2f} GiB."
            )

        for info in members:
            # Zip entries use POSIX paths even when created on Windows.
            member_path = PurePosixPath(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"ZIP chứa đường dẫn không an toàn: {info.filename}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError(f"ZIP chứa symbolic link không được hỗ trợ: {info.filename}")

            target = (destination / Path(*member_path.parts)).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"ZIP entry thoát khỏi thư mục đích: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)

    return _find_sample_root(destination)


def _discover_candidate_files(series_dir: Path) -> list[Path]:
    """Find files below a series directory, including nested/extensionless DICOMs."""
    if not series_dir.is_dir():
        return []
    return sorted(
        (
            path for path in series_dir.rglob("*")
            if path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(series_dir).parts)
            and path.name not in {"Thumbs.db", ".DS_Store"}
        ),
        key=lambda path: _natural_key(path.relative_to(series_dir).as_posix()),
    )


def _read_dicom_headers(series_dir: Path) -> tuple[list[SliceHeader], str]:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Thiếu pydicom. Cài bằng `python -m pip install 'pydicom>=3.0'`."
        ) from exc

    # Some Kaggle download/extraction tools introduce one or more extra directory
    # levels (or preserve nested subfolders) inside a SeriesInstanceUID folder.
    # Scan recursively rather than assuming every DICOM is an immediate child.
    # DICOM files can also be extensionless, so the suffix is not used as a
    # correctness requirement. pydicom will validate each candidate below.
    paths = _discover_candidate_files(series_dir)
    if not paths:
        return [], "no_dicom_files"

    raw_rows: list[dict[str, Any]] = []
    first_normal: np.ndarray | None = None
    for path in paths:
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True,
                specific_tags=HEADER_TAGS,
            )
            iop = getattr(ds, "ImageOrientationPatient", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            position = None
            if iop is not None and ipp is not None and len(iop) >= 6 and len(ipp) >= 3:
                row_cos = np.asarray(iop[:3], dtype=np.float64)
                col_cos = np.asarray(iop[3:6], dtype=np.float64)
                normal = np.cross(row_cos, col_cos)
                norm = float(np.linalg.norm(normal))
                if norm > 0:
                    normal /= norm
                    if first_normal is None:
                        first_normal = normal
                    position = float(np.dot(np.asarray(ipp[:3], dtype=np.float64), first_normal))

            raw_rows.append(
                {
                    "path": path,
                    "study_uid": str(getattr(ds, "StudyInstanceUID", "")),
                    "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
                    "sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
                    "instance": _safe_float(getattr(ds, "InstanceNumber", None)),
                    "position": _safe_float(position),
                    "rows": int(ds.Rows) if getattr(ds, "Rows", None) is not None else None,
                    "columns": int(ds.Columns) if getattr(ds, "Columns", None) is not None else None,
                    "modality": str(getattr(ds, "Modality", "")),
                    "header_error": "",
                }
            )
        except Exception as exc:
            raw_rows.append(
                {
                    "path": path,
                    "study_uid": "",
                    "series_uid": "",
                    "sop_uid": "",
                    "instance": None,
                    "position": None,
                    "rows": None,
                    "columns": None,
                    "modality": "",
                    "header_error": f"{type(exc).__name__}: {exc}",
                }
            )

    n = len(raw_rows)
    n_position = sum(row["position"] is not None for row in raw_rows)
    n_instance = sum(row["instance"] is not None for row in raw_rows)
    if n_position >= max(2, int(0.7 * n)):
        raw_rows.sort(
            key=lambda row: (
                math.inf if row["position"] is None else row["position"],
                row["path"].relative_to(series_dir).as_posix(),
            )
        )
        sort_mode = "ImagePositionPatient_projected"
    elif n_instance >= max(2, int(0.7 * n)):
        raw_rows.sort(
            key=lambda row: (
                math.inf if row["instance"] is None else row["instance"],
                row["path"].relative_to(series_dir).as_posix(),
            )
        )
        sort_mode = "InstanceNumber"
    else:
        raw_rows.sort(
            key=lambda row: _natural_key(row["path"].relative_to(series_dir).as_posix())
        )
        sort_mode = "filename_fallback"

    headers = [
        SliceHeader(
            path=row["path"],
            filename=row["path"].relative_to(series_dir).as_posix(),
            study_uid=row["study_uid"],
            series_uid=row["series_uid"],
            sop_uid=row["sop_uid"],
            instance_number=row["instance"],
            projected_position_mm=row["position"],
            rows=row["rows"],
            columns=row["columns"],
            modality=row["modality"],
            header_error=row["header_error"],
            sort_index=index,
        )
        for index, row in enumerate(raw_rows)
    ]
    return headers, sort_mode


def _validate_metadata(root: Path, study_uid: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    metadata_path = root / "series_metadata.csv"
    selection_path = root / "series_selection.csv"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "rsna-knee-local-demo-sample":
        raise ValueError(f"Định dạng sample không được hỗ trợ: {manifest.get('format')!r}")
    if int(manifest.get("format_version", 0)) != 1:
        raise ValueError(f"format_version không hỗ trợ: {manifest.get('format_version')}")

    source_uid = str(manifest.get("source", {}).get("study_instance_uid", ""))
    if source_uid != study_uid:
        raise ValueError(
            f"Study UID trong manifest ({source_uid}) không khớp DICOM folder ({study_uid})."
        )

    metadata = pd.read_csv(metadata_path, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    missing = REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"series_metadata.csv thiếu cột: {sorted(missing)}")
    metadata["StudyInstanceUID"] = metadata["StudyInstanceUID"].astype(str)
    metadata["SeriesInstanceUID"] = metadata["SeriesInstanceUID"].astype(str)
    if metadata.empty:
        raise ValueError("series_metadata.csv không có series nào.")
    if metadata["SeriesInstanceUID"].duplicated().any():
        duplicates = metadata.loc[
            metadata["SeriesInstanceUID"].duplicated(keep=False), "SeriesInstanceUID"
        ].astype(str).tolist()
        raise ValueError(f"SeriesInstanceUID bị trùng trong metadata: {duplicates[:10]}")
    if set(metadata["StudyInstanceUID"]) != {study_uid}:
        raise ValueError("series_metadata.csv chứa StudyInstanceUID khác study trong manifest.")

    selection = pd.read_csv(
        selection_path,
        dtype={"SeriesInstanceUID": str},
        keep_default_na=False,
    )
    missing = REQUIRED_SELECTION_COLUMNS - set(selection.columns)
    if missing:
        raise ValueError(f"series_selection.csv thiếu cột: {sorted(missing)}")
    selection["SeriesInstanceUID"] = selection["SeriesInstanceUID"].fillna("").astype(str).str.strip()
    nonempty_selection = set(selection.loc[selection["SeriesInstanceUID"].ne(""), "SeriesInstanceUID"])
    metadata_uids = set(metadata["SeriesInstanceUID"].astype(str))
    unknown_selection = nonempty_selection - metadata_uids
    if unknown_selection:
        raise ValueError(
            "series_selection.csv tham chiếu series không có trong metadata: "
            f"{sorted(unknown_selection)}"
        )

    return metadata, selection, manifest


def load_demo_sample(
    source: str | Path,
    *,
    extract_dir: str | Path | None = None,
) -> DemoStudy:
    """Load the exported sample from a ZIP or extracted directory.

    Parameters
    ----------
    source:
        ``demo_sample.zip`` or the directory containing ``manifest.json``.
    extract_dir:
        Destination used only when ``source`` is a ZIP. By default a sibling
        ``<zip-stem>_extracted`` directory is created.
    """
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Không tìm thấy demo sample: {source_path}")

    extracted_from: Path | None = None
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        target = (
            Path(extract_dir).expanduser().resolve()
            if extract_dir is not None
            else source_path.parent / f"{source_path.stem}_extracted"
        )
        sample_root = _safe_extract_zip(source_path, target)
        extracted_from = source_path
    elif source_path.is_dir():
        sample_root = _find_sample_root(source_path)
    else:
        raise ValueError("source phải là file .zip hoặc thư mục demo_sample đã giải nén.")

    dicom_root = sample_root / "dicom"
    if not dicom_root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dicom/: {dicom_root}")

    study_dirs = [path for path in dicom_root.iterdir() if path.is_dir()]
    if len(study_dirs) != 1:
        raise ValueError(
            f"Sample cần đúng một thư mục study dưới dicom/, tìm thấy {len(study_dirs)}."
        )
    study_dir = study_dirs[0]
    study_uid = study_dir.name

    metadata, selection, manifest = _validate_metadata(sample_root, study_uid)
    manifest_series = {
        str(item.get("series_instance_uid", ""))
        for item in manifest.get("dicom", {}).get("series", [])
    }
    metadata_series = set(metadata["SeriesInstanceUID"].astype(str))
    directory_series = {path.name for path in study_dir.iterdir() if path.is_dir()}
    if manifest_series != metadata_series:
        raise ValueError(
            "Danh sách series trong manifest không khớp series_metadata.csv. "
            f"manifest={sorted(manifest_series)}, metadata={sorted(metadata_series)}"
        )
    if directory_series != metadata_series:
        raise ValueError(
            "Thư mục DICOM không khớp metadata. "
            f"Thiếu={sorted(metadata_series - directory_series)}, "
            f"thừa={sorted(directory_series - metadata_series)}"
        )

    selected_for: dict[str, list[str]] = {}
    for _, row in selection.iterrows():
        uid = str(row["SeriesInstanceUID"]).strip()
        if uid:
            selected_for.setdefault(uid, []).append(f"{row['model']}:{row['role_or_plane']}")

    series: dict[str, DicomSeries] = {}
    metadata_by_uid = metadata.set_index("SeriesInstanceUID", drop=False)
    for series_uid in sorted(metadata_series):
        series_dir = study_dir / series_uid
        headers, sort_mode = _read_dicom_headers(series_dir)
        if not headers:
            entries = []
            try:
                for child in list(series_dir.iterdir())[:12]:
                    if child.is_dir():
                        nested_count = sum(1 for item in child.rglob("*") if item.is_file())
                        entries.append(f"{child.name}/ ({nested_count} files below)")
                    else:
                        entries.append(f"{child.name} ({child.stat().st_size} bytes)")
            except OSError as exc:
                entries.append(f"<cannot list directory: {exc}>")
            detail = "; ".join(entries) if entries else "<empty directory>"
            raise ValueError(
                f"Series {series_uid} không có file DICOM. "
                f"Đã quét đệ quy: {series_dir}. Nội dung cấp đầu: {detail}"
            )

        warnings: list[str] = []
        if any(item.header_error for item in headers):
            bad = sum(bool(item.header_error) for item in headers)
            warnings.append(f"{bad}/{len(headers)} file không đọc được header")

        bad_study = [
            item.filename for item in headers
            if item.study_uid and item.study_uid != study_uid
        ]
        bad_series = [
            item.filename for item in headers
            if item.series_uid and item.series_uid != series_uid
        ]
        if bad_study:
            raise ValueError(
                f"Series {series_uid} chứa file có StudyInstanceUID sai: {bad_study[:5]}"
            )
        if bad_series:
            raise ValueError(
                f"Thư mục {series_uid} chứa file có SeriesInstanceUID sai: {bad_series[:5]}"
            )

        sop_uids = [item.sop_uid for item in headers if item.sop_uid]
        if len(sop_uids) != len(set(sop_uids)):
            warnings.append("SOPInstanceUID bị trùng trong series")
        shapes = {
            (item.rows, item.columns)
            for item in headers
            if item.rows is not None and item.columns is not None
        }
        if len(shapes) > 1:
            warnings.append(f"DICOM slices có nhiều kích thước Rows/Columns: {sorted(shapes)}")
        if any(item.header_error for item in headers):
            # File with an unreadable header is retained in the ordered list, but is
            # expected to fail model preprocessing and be skipped there.
            pass

        row = metadata_by_uid.loc[series_uid]
        series[series_uid] = DicomSeries(
            series_uid=series_uid,
            directory=series_dir,
            metadata=row.to_dict(),
            files=headers,
            sort_mode=sort_mode,
            selected_for=selected_for.get(series_uid, []),
            warnings=warnings,
        )

    return DemoStudy(
        root=sample_root,
        study_uid=study_uid,
        manifest=manifest,
        metadata=metadata,
        selection=selection,
        series=series,
        extracted_from=extracted_from,
    )


def audit_demo_sample(
    study: DemoStudy,
    *,
    decode_pixels: bool = True,
) -> dict[str, Any]:
    """Return an audit report; optionally decode one central image per series."""
    issues: list[dict[str, str]] = []
    series_rows: list[dict[str, Any]] = []

    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Thiếu pydicom. Cài `python -m pip install 'pydicom>=3.0'`.") from exc

    expected_roles = ["SAG_FLUID", "COR_FLUID", "AX_FLUID", "SAG_NONFLUID", "COR_NONFLUID"]
    expected_planes = ["Sagittal", "Coronal", "Axial"]
    for model, expected in (("DINOv2", expected_roles), ("ViT-B/16", expected_planes)):
        model_rows = study.selection[
            study.selection["model"].astype(str).str.casefold().eq(model.casefold())
        ]
        missing = [
            key for key in expected
            if not model_rows["role_or_plane"].astype(str).eq(key).any()
            or not model_rows.loc[
                model_rows["role_or_plane"].astype(str).eq(key), "SeriesInstanceUID"
            ].astype(str).str.strip().ne("").any()
        ]
        if missing:
            issues.append(
                {"severity": "warning", "scope": model, "message": f"Thiếu series mapping: {missing}"}
            )

    for uid, item in study.series.items():
        pixel_status = "not_checked"
        pixel_shape: list[int] | None = None
        pixel_error = ""
        if decode_pixels:
            readable = [header for header in item.files if not header.header_error]
            if readable:
                probe = readable[len(readable) // 2]
                try:
                    ds = pydicom.dcmread(str(probe.path), force=True)
                    pixels = np.asarray(ds.pixel_array)
                    if pixels.size == 0:
                        raise ValueError("pixel_array rỗng")
                    pixel_status = "ok"
                    pixel_shape = [int(value) for value in pixels.shape]
                except Exception as exc:
                    pixel_status = "failed"
                    pixel_error = f"{type(exc).__name__}: {exc}"
            else:
                pixel_status = "failed"
                pixel_error = "Không có DICOM header hợp lệ để decode thử."

        if item.warnings:
            issues.append(
                {"severity": "warning", "scope": uid, "message": "; ".join(item.warnings)}
            )
        if pixel_status == "failed":
            issues.append(
                {"severity": "error", "scope": uid, "message": f"Pixel decode thất bại: {pixel_error}"}
            )

        manifest_row = next(
            (
                row for row in study.manifest.get("dicom", {}).get("series", [])
                if str(row.get("series_instance_uid", "")) == uid
            ),
            {},
        )
        expected_files = manifest_row.get("n_files")
        if expected_files is not None and int(expected_files) != item.n_files:
            issues.append(
                {
                    "severity": "error",
                    "scope": uid,
                    "message": f"Số DICOM thực tế {item.n_files} khác manifest {expected_files}.",
                }
            )

        series_rows.append(
            {
                "SeriesInstanceUID": uid,
                "plane": item.metadata.get("Anatomical_Plane", ""),
                "fluid_sensitive": item.metadata.get("Fluid_Sensitive", ""),
                "fat_suppression": item.metadata.get("Fat_Suppression", ""),
                "n_files": item.n_files,
                "valid_headers": item.usable_header_count,
                "sort_mode": item.sort_mode,
                "selected_for": ";".join(item.selected_for),
                "pixel_decode": pixel_status,
                "pixel_shape": pixel_shape,
            }
        )

    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    return {
        "status": "ok" if not errors else "failed",
        "study_instance_uid": study.study_uid,
        "sample_root": str(study.root),
        "series_count": len(study.series),
        "dicom_file_count": sum(item.n_files for item in study.series.values()),
        "dino_roles_selected": int(
            study.selection.loc[
                study.selection["model"].astype(str).str.casefold().eq("dinov2"),
                "SeriesInstanceUID",
            ].astype(str).str.strip().ne("").sum()
        ),
        "vit_planes_selected": int(
            study.selection.loc[
                study.selection["model"].astype(str).str.casefold().eq("vit-b/16"),
                "SeriesInstanceUID",
            ].astype(str).str.strip().ne("").sum()
        ),
        "series": series_rows,
        "issues": issues,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "pixel_decode_checked": bool(decode_pixels),
    }


def write_audit_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write audit report as UTF-8 JSON."""
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
