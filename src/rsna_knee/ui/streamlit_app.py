"""Streamlit dashboard for viewing one local RSNA knee demo study.

The app is deliberately read-only with respect to model inference: it reads the
Phase 3 ensemble outputs and displays the exported DICOM series. It never loads
model checkpoints or runs DINOv2/ViT forward passes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    load_prediction_bundle,
)
from rsna_knee.ui.dicom_display import read_dicom_preview  # noqa: E402


st.set_page_config(
    page_title="RSNA Knee MRI — Local Demo",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


STATUS_LABELS = {
    "weighted_ensemble": "Weighted ensemble (DINOv2 90% + ViT 10%)",
    "dinov2_only": "DINOv2-only fallback",
    "vit_only": "ViT-only fallback",
    "neutral_fallback_0.5": "Neutral fallback (0.5)",
}


def _sample_default() -> str:
    configured = os.environ.get("RSNA_DEMO_SAMPLE", "").strip()
    if configured and Path(configured).exists():
        return configured
    # Prefer the short path used during the Phase 1 DICOM audit. Deeply nested
    # Study/Series/SOP UID folders can push a file path over Windows MAX_PATH.
    candidates = [
        Path(r"C:\rsna_demo\demo_sample"),
        PROJECT_ROOT / "demo" / "input" / "demo_sample",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return str(candidates[0])


@st.cache_resource(show_spinner="Đang đọc metadata DICOM của study...")
def _load_study(sample_path: str):
    # Import the Phase 1 reader lazily so the prediction dashboard can still
    # open if the optional DICOM dependencies are unavailable.
    from rsna_knee.dicom.sample import load_demo_sample

    return load_demo_sample(sample_path)


@st.cache_data(show_spinner=False)
def _load_preview(path_text: str, modified_ns: int) -> np.ndarray:
    # modified_ns participates in the cache key so an updated file is re-read.
    del modified_ns
    return read_dicom_preview(path_text)


def _model_status_text(report: dict[str, Any], key: str) -> str:
    model = report.get("models", {}).get(key, {})
    valid = model.get("valid") is True
    status = str(model.get("status", "unknown"))
    return f"Hợp lệ · {status}" if valid else f"Không dùng trong blend · {status}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_prediction_section(bundle: PredictionBundle) -> None:
    report = bundle.report
    blend_status = str(report.get("blend_status", ""))
    models = report.get("models", {})
    dino_valid = bool(models.get("dinov2", {}).get("valid", False))
    vit_valid = bool(models.get("vit_b16", {}).get("valid", False))

    st.subheader("Tổng quan dự đoán")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("StudyInstanceUID", f"…{bundle.study_uid[-12:]}")
    m2.metric("DINOv2", "Valid" if dino_valid else "Fallback / unusable")
    m3.metric("ViT-B/16", "Valid" if vit_valid else "Fallback / unusable")
    m4.metric("Blend status", blend_status.replace("_", " "))

    if blend_status == "neutral_fallback_0.5":
        st.warning(
            "Cả hai nhánh đều không có dự đoán hợp lệ. Dashboard đang hiển thị giá trị trung tính 0.5, "
            "không phải kết quả suy luận của model."
        )
    elif blend_status in {"dinov2_only", "vit_only"}:
        st.info(
            "Một nhánh không được dùng trong blend. Kết quả cuối lấy trực tiếp từ nhánh còn hợp lệ; "
            "trọng số hiệu lực được ghi trong report."
        )
    else:
        st.caption(STATUS_LABELS.get(blend_status, blend_status))

    st.caption(
        f"Study UID: `{bundle.study_uid}` · Output directory: `{bundle.output_dir}`"
    )

    st.subheader("Xác suất theo từng nhãn")
    threshold = st.slider(
        "Mốc tham chiếu để sắp xếp / tô nổi bật xác suất (không phải ngưỡng chẩn đoán)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        format="%.2f",
    )

    chart_table = pd.DataFrame(
        {
            "Target": bundle.targets,
            "DINOv2": bundle.dinov2,
            "ViT-B/16": bundle.vit,
            "Ensemble": bundle.ensemble,
        }
    )
    chart_long = chart_table.melt(
        id_vars="Target", var_name="Model", value_name="Probability"
    )
    chart_long["Probability (%)"] = chart_long["Probability"] * 100.0
    fig = px.bar(
        chart_long,
        x="Target",
        y="Probability (%)",
        color="Model",
        barmode="group",
        category_orders={"Target": list(bundle.targets)},
        hover_data={"Probability (%)": ":.2f", "Probability": False},
        title="So sánh xác suất của hai model và ensemble",
    )
    fig.update_layout(
        yaxis_range=[0, 100],
        xaxis_title="",
        yaxis_title="Probability (%)",
        legend_title="",
        margin=dict(l=10, r=10, t=55, b=10),
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame(
        {
            "Target": bundle.targets,
            "DINOv2": bundle.dinov2,
            "ViT-B/16": bundle.vit,
            "Ensemble": bundle.ensemble,
        }
    )
    table["|DINOv2 − ViT| (pp)"] = np.abs(bundle.dinov2 - bundle.vit) * 100.0
    table["So với mốc"] = np.where(
        bundle.ensemble >= threshold,
        f"≥ {threshold:.2f}",
        f"< {threshold:.2f}",
    )
    display_table = table.copy()
    for column in ("DINOv2", "ViT-B/16", "Ensemble"):
        display_table[column] = display_table[column].map(_format_percent)
    display_table["|DINOv2 − ViT| (pp)"] = display_table[
        "|DINOv2 − ViT| (pp)"
    ].map(lambda x: f"{x:.1f}")
    st.dataframe(display_table, hide_index=True, use_container_width=True)

    st.markdown("**Trạng thái và nguồn kết quả**")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.write("**DINOv2**")
        st.write(_model_status_text(report, "dinov2"))
        st.caption(
            f"Fold count: {models.get('dinov2', {}).get('metadata', {}).get('fold_count', 'n/a')}"
        )
    with status_col2:
        st.write("**ViT-B/16**")
        st.write(_model_status_text(report, "vit_b16"))
        st.caption(
            f"Valid slices: {models.get('vit_b16', {}).get('metadata', {}).get('valid_slices', 'n/a')}"
        )

    with st.expander("Xem report provenance của Phase 3"):
        st.json(report, expanded=False)


def _series_label(series: Any) -> str:
    metadata = series.metadata
    plane = str(metadata.get("Anatomical_Plane", "Unknown plane"))
    fluid = str(metadata.get("Fluid_Sensitive", "?"))
    fat = str(metadata.get("Fat_Suppression", "?"))
    roles = ", ".join(series.selected_for) if series.selected_for else "not mapped"
    return (
        f"{plane} · Fluid={fluid} · Fat suppression={fat} · "
        f"{series.n_files} slices · {roles} · UID …{series.series_uid[-8:]}"
    )


def _render_mri_viewer(sample_path: str, expected_study_uid: str) -> None:
    st.subheader("MRI DICOM viewer")
    if not sample_path.strip():
        st.info("Nhập đường dẫn tới thư mục demo_sample để xem MRI.")
        return
    root = Path(sample_path).expanduser()
    if not root.exists():
        st.warning(f"Không tìm thấy sample path: `{root}`")
        return

    try:
        study = _load_study(str(root))
    except Exception as exc:
        st.warning(f"Không tải được DICOM sample: {type(exc).__name__}: {exc}")
        st.caption(
            "Dashboard dự đoán vẫn hoạt động độc lập; kiểm tra lại sample path và dependencies DICOM."
        )
        return

    if study.study_uid != expected_study_uid:
        st.error(
            "StudyInstanceUID của DICOM sample không khớp prediction artifacts. "
            f"DICOM={study.study_uid} · Predictions={expected_study_uid}."
        )
        return

    summary = study.series_summary()
    st.caption(
        f"{len(study.series)} selected series · "
        f"{sum(item.n_files for item in study.series.values())} DICOM files · "
        f"Study UID khớp với prediction artifacts"
    )

    series_list = list(study.series.values())
    if not series_list:
        st.warning("Sample không có series để hiển thị.")
        return

    series_by_label = {_series_label(item): item for item in series_list}
    labels = list(series_by_label)
    selected_label = st.selectbox("Chọn series / plane / role", labels)
    selected_series = series_by_label[selected_label]
    readable_files = [item for item in selected_series.files if not item.header_error]
    if not readable_files:
        st.error("Series không có DICOM header hợp lệ để hiển thị.")
        return

    default_index = len(readable_files) // 2
    slice_index = st.slider(
        "Slice index (theo thứ tự đã audit)",
        min_value=0,
        max_value=len(readable_files) - 1,
        value=default_index,
        step=1,
        key=f"slice_{selected_series.series_uid}",
    )
    dicom_header = readable_files[slice_index]

    image_col, meta_col = st.columns([1.35, 1])
    with image_col:
        try:
            modified_ns = dicom_header.path.stat().st_mtime_ns
            preview = _load_preview(str(dicom_header.path), modified_ns)
            st.image(
                preview,
                caption=(
                    f"{dicom_header.filename} · slice {slice_index + 1}/{len(readable_files)}"
                ),
                use_container_width=True,
                clamp=True,
            )
        except Exception as exc:
            st.error(f"Không decode được pixel data: {type(exc).__name__}: {exc}")
    with meta_col:
        st.markdown("**Thông tin series**")
        info = {
            "Anatomical plane": selected_series.metadata.get("Anatomical_Plane", ""),
            "Fluid sensitive": selected_series.metadata.get("Fluid_Sensitive", ""),
            "Fat suppression": selected_series.metadata.get("Fat_Suppression", ""),
            "Number of slices": selected_series.n_files,
            "Valid headers": selected_series.usable_header_count,
            "Sort mode": selected_series.sort_mode,
            "Selected for": selected_series.selected_for,
            "SeriesInstanceUID": selected_series.series_uid,
            "Filename": dicom_header.filename,
            "InstanceNumber": dicom_header.instance_number,
            "Projected position (mm)": dicom_header.projected_position_mm,
        }
        st.json(info, expanded=True)
        if selected_series.warnings:
            st.warning(" · ".join(selected_series.warnings))

    with st.expander("Metadata của toàn bộ selected series"):
        st.dataframe(summary, hide_index=True, use_container_width=True)


def main() -> None:
    st.title("RSNA Knee MRI — Local Demo")
    st.caption(
        "Study-level predictions · DINOv2 + ViT-B/16 ensemble · Read-only visualization"
    )

    with st.sidebar:
        st.header("Đường dẫn dữ liệu")
        output_default = PROJECT_ROOT / "demo" / "output" / "ensemble"
        output_text = st.text_input(
            "Phase 3 ensemble output directory",
            value=str(output_default),
            help="Thư mục chứa ensemble_prediction.csv, ensemble_prediction_long.csv và ensemble_report.json.",
        )
        sample_text = st.text_input(
            "Extracted DICOM sample directory",
            value=_sample_default(),
            help=(
                "Thư mục chứa manifest.json, series_metadata.csv, series_selection.csv và dicom/. "
                "Ưu tiên đường dẫn ngắn như C:\\rsna_demo\\demo_sample để tránh giới hạn path của Windows."
            ),
        )
        if sample_text.strip() and len(str(Path(sample_text).expanduser())) > 40:
            st.warning(
                "Đường dẫn sample đang khá dài. DICOM được lồng qua Study UID, Series UID và "
                "tên file nên full path có thể vượt giới hạn MAX_PATH truyền thống của Windows. "
                "Ưu tiên C:\\rsna_demo\\demo_sample nếu thư mục này đã được audit."
            )
        show_mri = st.checkbox("Hiển thị MRI DICOM viewer", value=True)
        st.divider()
        st.caption(f"Project root: `{PROJECT_ROOT}`")
        st.caption("Dashboard không tải checkpoint hoặc chạy model inference.")

    try:
        bundle = load_prediction_bundle(output_text)
    except (DashboardDataError, OSError, ValueError) as exc:
        st.error(f"Không thể tải prediction artifacts: {exc}")
        st.info(
            "Hãy chạy thành công Notebook 14 (Phase 3), sau đó kiểm tra lại đường dẫn ensemble output."
        )
        st.stop()

    _render_prediction_section(bundle)

    if show_mri:
        st.divider()
        _render_mri_viewer(sample_text, bundle.study_uid)

    st.divider()
    st.caption(
        "Lưu ý: đây là xác suất đầu ra của mô hình cho một sample thuộc gold_validation, "
        "không phải chẩn đoán y khoa, không phải xác suất bệnh đã được hiệu chuẩn, "
        "và không phải một ước lượng validation độc lập hay điểm số Kaggle mới. "
        "Ảnh MRI trong viewer chỉ dùng cho quan sát; windowing hiển thị không thay thế preprocessing của model."
    )


if __name__ == "__main__":
    main()
