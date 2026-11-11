"""Layperson-oriented Streamlit UI for one RSNA knee MRI study.

The dashboard reads the Phase 3 ensemble output and local DICOM sample only.
It never loads model checkpoints or runs DINOv2/ViT inference.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.express as px
import streamlit as st


def _project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return cwd


PROJECT_ROOT = _project_root()
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rsna_knee.ui.dashboard_data import (  # noqa: E402
    DashboardDataError,
    PredictionBundle,
    TARGETS,
    find_prediction_dir,
    load_prediction_bundle,
)
from rsna_knee.ui.dicom_display import read_dicom_thumbnail  # noqa: E402
from rsna_knee.dicom.paths import long_path  # noqa: E402
from rsna_knee.ui.study_context import (  # noqa: E402
    PatientInfo,
    StudyContextError,
    load_radiology_report,
    read_patient_info,
)

st.set_page_config(
    page_title="RSNA Knee MRI — Demo",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


PLANE_INFO = (
    (
        "sagittal",
        "Mặt phẳng dọc (Sagittal)",
        "Ảnh nhìn dọc theo đầu gối, giúp quan sát cấu trúc phía trước và phía sau.",
    ),
    (
        "coronal",
        "Mặt phẳng trán (Coronal)",
        "Ảnh nhìn từ phía trước, giúp quan sát phần trong và phần ngoài của khớp gối.",
    ),
    (
        "axial",
        "Mặt phẳng ngang (Axial)",
        "Ảnh cắt ngang qua khớp gối.",
    ),
)

TARGET_DISPLAY: dict[str, tuple[str, str, str]] = {
    "ACL": (
        "Tổn thương dây chằng chéo trước",
        "Dây chằng quan trọng giúp giữ vững khớp gối khi xoay và đổi hướng.",
        "Dây chằng",
    ),
    "MCL": (
        "Tổn thương dây chằng bên trong",
        "Dây chằng nằm ở phía trong đầu gối, giúp chống lực làm gối lệch vào trong.",
        "Dây chằng",
    ),
    "Medial Meniscus": (
        "Tổn thương sụn chêm trong",
        "Sụn đệm hình bán nguyệt ở phía trong khớp gối.",
        "Sụn chêm",
    ),
    "Lateral Meniscus": (
        "Tổn thương sụn chêm ngoài",
        "Sụn đệm hình bán nguyệt ở phía ngoài khớp gối.",
        "Sụn chêm",
    ),
    "Medial OA": (
        "Thoái hóa khoang khớp trong",
        "Dấu hiệu hao mòn ở phần khớp nằm phía trong đầu gối.",
        "Thoái hóa khớp",
    ),
    "Lateral OA": (
        "Thoái hóa khoang khớp ngoài",
        "Dấu hiệu hao mòn ở phần khớp nằm phía ngoài đầu gối.",
        "Thoái hóa khớp",
    ),
    "PF OA": (
        "Thoái hóa khớp bánh chè – xương đùi",
        "Dấu hiệu hao mòn tại mặt khớp giữa xương bánh chè và xương đùi.",
        "Thoái hóa khớp",
    ),
    "Effusion": (
        "Tràn dịch khớp gối",
        "Lượng dịch trong khớp tăng nhiều hơn bình thường.",
        "Dấu hiệu khác",
    ),
    "Synovitis": (
        "Viêm màng hoạt dịch",
        "Tình trạng viêm của lớp mô lót bên trong bao khớp.",
        "Dấu hiệu khác",
    ),
    "Baker's": (
        "Nang Baker",
        "Túi dịch thường hình thành ở vùng phía sau khớp gối.",
        "Dấu hiệu khác",
    ),
    "Contusion": (
        "Dập xương (bone bruise)",
        "Tổn thương bên trong xương do va đập, không nhất thiết đồng nghĩa với gãy xương.",
        "Dấu hiệu khác",
    ),
    "Fracture": (
        "Gãy xương",
        "Tổn thương làm mất tính liên tục của xương.",
        "Dấu hiệu khác",
    ),
}

BLEND_LABELS = {
    "weighted_ensemble": "Đã kết hợp hai mô hình",
    "dinov2_only": "Chỉ sử dụng mô hình DINOv2",
    "vit_only": "Chỉ sử dụng mô hình ViT-B/16",
    "neutral_fallback_0.5": "Không có mô hình hợp lệ — dùng giá trị trung tính",
}


def _sample_default() -> str:
    configured = os.environ.get("RSNA_DEMO_SAMPLE", "").strip()
    if configured and long_path(configured).exists():
        return configured
    candidates = [
        PROJECT_ROOT / "demo" / "input" / "demo_sample",
        PROJECT_ROOT / "demo" / "input",
        Path(r"C:\rsna_demo\demo_sample"),
    ]
    for candidate in candidates:
        if long_path(candidate).is_dir():
            return str(candidate)
    return str(candidates[-1])


def _report_default(sample_path: str) -> str:
    root = Path(sample_path).expanduser()
    for name in ("radiology_report.txt", "radiology_report.md", "doctor_report.txt", "doctor_report.md"):
        candidate = root / name
        if candidate.is_file():
            return str(candidate)
    return ""


@st.cache_resource(show_spinner="Đang đọc thông tin study và thứ tự lát MRI...")
def _load_study(sample_path: str):
    from rsna_knee.dicom.sample import load_demo_sample

    return load_demo_sample(sample_path)


@st.cache_data(show_spinner=False, ttl=5)
def _discover_sample_paths(sample_root: str) -> list[str]:
    from rsna_knee.dicom.sample import discover_demo_samples

    return [str(path) for path in discover_demo_samples(sample_root)]


def _sample_label(sample_path: str) -> str:
    """Create a short, distinguishable label for a study package."""
    path = long_path(sample_path)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        study_uid = str(manifest.get("source", {}).get("study_instance_uid", ""))
    except (OSError, json.JSONDecodeError, AttributeError):
        study_uid = ""
    suffix = f" · …{study_uid[-10:]}" if study_uid else ""
    return f"{path.name}{suffix}"


@st.cache_data(show_spinner=False, max_entries=96)
def _load_thumbnail(path_text: str, modified_ns: int, max_side: int = 320) -> np.ndarray:
    # mtime participates in the cache key so replaced DICOM files are re-read.
    del modified_ns
    return read_dicom_thumbnail(path_text, max_side=max_side)


def _plane_key(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if "sag" in text:
        return "sagittal"
    if "cor" in text:
        return "coronal"
    if "axi" in text or "trans" in text:
        return "axial"
    return "unknown"


def _render_patient_info(info: PatientInfo) -> None:
    st.subheader("Thông tin người bệnh")
    cols = st.columns(5)
    fields = (
        ("Tuổi", info.age),
        ("Giới tính", info.sex),
        ("Bên gối", info.knee_side),
        ("Ngày chụp", info.study_date),
        ("Vùng chụp", info.body_part),
    )
    for col, (label, value) in zip(cols, fields):
        with col:
            st.metric(label, value)
    st.caption(
        f"Phương thức chụp: {info.modality}. Thông tin được đọc từ metadata DICOM; "
        "dashboard không hiển thị tên hoặc mã định danh cá nhân. Trường nào không có trong dữ liệu sẽ được ghi rõ."
    )


def _collect_plane_images(study: Any, plane: str) -> tuple[list[dict[str, Any]], int]:
    series_rows = []
    for series in study.series.values():
        if _plane_key(series.metadata.get("Anatomical_Plane", "")) == plane:
            series_rows.append(series)
    series_rows.sort(key=lambda item: item.series_uid)

    tiles: list[dict[str, Any]] = []
    for series_number, series in enumerate(series_rows, start=1):
        readable = [item for item in series.files if not item.header_error]
        for index, header in enumerate(readable, start=1):
            tiles.append(
                {
                    "path": header.path,
                    "series_number": series_number,
                    "slice_number": index,
                    "series_slice_count": len(readable),
                }
            )
    return tiles, len(series_rows)


def _render_plane_gallery(study: Any, *, plane: str, title: str, description: str) -> None:
    tiles, series_count = _collect_plane_images(study, plane)
    st.subheader(title)
    if not tiles:
        st.info("Sample không có ảnh ở mặt phẳng này.")
        return

    st.caption(f"{description} · {series_count} chuỗi MRI · {len(tiles)} ảnh")
    page_size = 12
    page_count = (len(tiles) + page_size - 1) // page_size
    page = 1
    if page_count > 1:
        page = st.slider(
            "Trang ảnh",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key=f"gallery_page_{plane}",
        )
    start = (page - 1) * page_size
    page_tiles = tiles[start : start + page_size]
    columns = st.columns(4)
    for index, tile in enumerate(page_tiles):
        col = columns[index % 4]
        path = tile["path"]
        with col:
            try:
                mtime_ns = path.stat().st_mtime_ns
                image = _load_thumbnail(str(path), mtime_ns, 320)
                st.image(
                    image,
                    caption=(
                        f"Chuỗi {tile['series_number']} · "
                        f"lát {tile['slice_number']}/{tile['series_slice_count']}"
                    ),
                    use_container_width=True,
                    clamp=True,
                )
            except Exception as exc:
                st.warning(f"Không đọc được một lát ảnh: {type(exc).__name__}: {exc}")
    st.caption(f"Đang hiển thị ảnh {start + 1}–{start + len(page_tiles)} / {len(tiles)}.")


def _render_mri_galleries(study: Any) -> None:
    st.header("Hình ảnh MRI theo ba hướng chụp")
    st.caption(
        "Mỗi ô là một lát cắt MRI. Dùng thanh chọn trang trong từng gallery để xem các lát tiếp theo."
    )
    for plane, title, description in PLANE_INFO:
        _render_plane_gallery(
            study,
            plane=plane,
            title=title,
            description=description,
        )


def _render_radiology_report(sample_root: Path, study_uid: str, report_path: str) -> None:
    st.header("Báo cáo bác sĩ")
    try:
        report = load_radiology_report(
            sample_root,
            study_uid,
            report_path=report_path or None,
        )
    except StudyContextError as exc:
        st.error(str(exc))
        return

    if report is None:
        st.info(
            "Sample hiện chưa có file báo cáo bác sĩ. Báo cáo gốc nằm trong cột `Report` của "
            "`train.csv`; hãy chạy notebook `15_export_demo_radiology_report.ipynb` "
            "và đặt `radiology_report.txt` vào thư mục demo_sample."
        )
        return

    source_text = report.source or "Báo cáo văn bản đi kèm sample"
    verification_text = "Đã đối chiếu mã study" if report.uid_verified else "Không có mã study trong file để đối chiếu"
    st.caption(f"Nguồn: {source_text} · {verification_text}")
    with st.container(border=True):
        st.markdown(report.text, unsafe_allow_html=False)
    st.caption(
        "Nội dung được giữ nguyên theo báo cáo nguồn; dashboard chỉ chuẩn hóa khoảng trắng và xuống dòng, "
        "không tự dịch, diễn giải hay thêm kết luận y khoa."
    )


def _render_predictions(bundle: PredictionBundle) -> None:
    st.header("Kết quả dự đoán 12 dấu hiệu")
    blend_status = str(bundle.report.get("blend_status", ""))
    blend_label = BLEND_LABELS.get(blend_status, "Trạng thái kết hợp không xác định")
    st.caption(blend_label)

    if blend_status == "neutral_fallback_0.5":
        st.warning(
            "Hai mô hình đều không tạo được dự đoán hợp lệ. Các giá trị 50% dưới đây là giá trị trung tính "
            "được dùng để hoàn thiện output, không phải kết quả suy luận của mô hình."
        )
    elif blend_status in {"dinov2_only", "vit_only"}:
        st.info("Một trong hai mô hình không khả dụng; kết quả bên dưới chỉ lấy từ mô hình còn hợp lệ.")

    display_rows = []
    for target, probability in zip(bundle.targets, bundle.ensemble):
        name, description, group = TARGET_DISPLAY[target]
        display_rows.append((group, target, name, description, float(probability)))

    group_order = ("Dây chằng", "Sụn chêm", "Thoái hóa khớp", "Dấu hiệu khác")
    for group in group_order:
        rows = [row for row in display_rows if row[0] == group]
        if not rows:
            continue
        st.subheader(group)
        for _, target, name, description, probability in rows:
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{name}**")
                st.caption(description)
                st.progress(float(np.clip(probability, 0.0, 1.0)))
            with right:
                st.metric("Điểm mô hình", f"{probability * 100:.1f}%")

    st.caption(
        "Cách đọc: phần trăm là điểm đầu ra của mô hình cho từng nhãn, không phải tỷ lệ chắc chắn người bệnh mắc bệnh "
        "và chưa được hiệu chuẩn thành nguy cơ lâm sàng. Không dùng riêng các con số này để chẩn đoán hoặc quyết định điều trị."
    )


def _render_technical_details(bundle: PredictionBundle, study: Any | None) -> None:
    with st.expander("Thông tin kỹ thuật dành cho người phát triển"):
        models = bundle.report.get("models", {})
        blend = bundle.report.get("blend", {})
        effective = blend.get("effective_weights", {}) if isinstance(blend, dict) else {}
        dino = models.get("dinov2", {}) if isinstance(models, dict) else {}
        vit = models.get("vit_b16", {}) if isinstance(models, dict) else {}
        dino_meta = dino.get("metadata", {}) if isinstance(dino, dict) else {}
        vit_meta = vit.get("metadata", {}) if isinstance(vit, dict) else {}

        status_labels = {
            True: "Được sử dụng",
            False: "Không được sử dụng",
        }
        st.write(f"**StudyInstanceUID:** `{bundle.study_uid}`")
        st.write(f"**DINOv2:** {status_labels.get(dino.get('valid'), 'Không rõ')}")
        st.write(f"**ViT-B/16:** {status_labels.get(vit.get('valid'), 'Không rõ')}")
        st.write(
            f"**Trọng số hiệu lực:** DINOv2 {float(effective.get('dinov2', 0.0)):.0%} · "
            f"ViT-B/16 {float(effective.get('vit_b16', 0.0)):.0%}"
        )
        if dino_meta:
            st.write(f"**Số fold DINOv2:** {dino_meta.get('fold_count', 'Không có dữ liệu')}")
        if vit_meta:
            st.write(f"**Số lát hợp lệ của ViT:** {vit_meta.get('valid_slices', 'Không có dữ liệu')}")
        if study is not None:
            st.write(f"**Số chuỗi MRI:** {len(study.series)}")
            st.write(
                f"**Số file DICOM:** {sum(item.n_files for item in study.series.values())}"
            )
        st.write(f"**Thư mục output:** `{bundle.output_dir}`")


def main() -> None:
    st.title("RSNA Knee MRI — Trình xem study")
    st.caption("Thông tin người bệnh · Ảnh MRI ba hướng · Báo cáo bác sĩ · Điểm dự đoán của mô hình")

    output_default = PROJECT_ROOT / "demo" / "output" / "ensemble"
    selected_sample: str | None = None
    sample_paths: list[str] = []
    with st.sidebar:
        st.header("Trình xem study")
        st.caption("Chọn study để xem ảnh MRI và điểm dự đoán đã lưu.")
        with st.expander("Thư viện study", expanded=False):
            sample_root = st.text_input(
                "Thư mục chứa các study",
                value=_sample_default(),
                help=(
                    "Chọn một gói đã xuất hoặc thư mục cha chứa nhiều gói. "
                    "Mỗi gói cần manifest.json, series_metadata.csv, series_selection.csv và thư mục dicom/."
                ),
            )
            if st.button("Làm mới danh sách study", use_container_width=True):
                _discover_sample_paths.clear()
                st.rerun()

        try:
            source = sample_root.strip()
            if source:
                sample_paths = _discover_sample_paths(source)
        except (OSError, ValueError) as exc:
            st.error(f"Không thể đọc thư mục này: {exc}")

        if sample_paths:
            selected_sample = st.selectbox(
                "Chọn study",
                options=sample_paths,
                format_func=_sample_label,
                help="Tên thư mục và phần cuối của Study UID giúp phân biệt các study.",
            )
            st.caption(f"Tìm thấy {len(sample_paths)} gói study.")
        else:
            st.info(
                "Chưa tìm thấy gói study. Hãy thêm một gói đã xuất đầy đủ vào thư viện, "
                "sau đó nhấn Làm mới danh sách study."
            )

        with st.expander("Cài đặt nâng cao", expanded=False):
            output_text = st.text_input(
                "Thư mục kết quả dự đoán Phase 3",
                value=str(output_default),
                help=(
                    "Ứng dụng ghép kết quả với study đã chọn bằng StudyInstanceUID. "
                    "Thông thường có thể dùng thư mục mặc định demo/output/ensemble."
                ),
            )
            report_text = st.text_input(
                "Đường dẫn báo cáo bác sĩ (không bắt buộc)",
                value="",
                help="Để trống để dùng báo cáo đi kèm study đã chọn, nếu có.",
            )
        st.divider()
        st.caption("Trình xem đọc file DICOM và kết quả dự đoán đã lưu; không chạy suy luận mô hình.")

    study = None
    if selected_sample:
        try:
            study = _load_study(selected_sample)
        except Exception as exc:
            st.error(f"Không thể mở study đã chọn: {type(exc).__name__}: {exc}")

    bundle = None
    if study is not None:
        try:
            prediction_dir = find_prediction_dir(output_text, study.study_uid)
            bundle = load_prediction_bundle(prediction_dir)
        except (DashboardDataError, OSError, ValueError) as exc:
            st.info(
                f"Không tìm thấy kết quả dự đoán hợp lệ cho study {study.study_uid}. "
                f"Bạn vẫn có thể xem ảnh MRI và thông tin study. Chi tiết: {exc}"
            )

    if study is not None:
        st.subheader(_sample_label(selected_sample))
        st.caption(f"StudyInstanceUID: `{study.study_uid}`")
        try:
            _render_patient_info(read_patient_info(study))
        except Exception as exc:
            st.warning(f"Không đọc được một số metadata người bệnh: {type(exc).__name__}: {exc}")
        _render_mri_galleries(study)
        _render_radiology_report(study.root, study.study_uid, report_text)

    if bundle is not None:
        st.divider()
        _render_predictions(bundle)
        _render_technical_details(bundle, study)
    elif study is None:
        st.info(
            "Chọn một study ở thanh bên để bắt đầu. Nếu danh sách trống, hãy chọn thư mục thư viện trong mục "
            "Thư viện study rồi làm mới danh sách."
        )

    st.divider()
    st.caption(
        "Trình xem hiển thị ảnh MRI của study đã chọn và điểm mô hình đã lưu nếu có. "
        "Điểm số không phải chẩn đoán hoặc nguy cơ lâm sàng đã hiệu chuẩn. Ảnh thu nhỏ MRI được chỉnh cửa sổ để hiển thị "
        "và không phải dữ liệu đã qua tiền xử lý đầu vào của mô hình."
    )

if __name__ == "__main__":
    main()
