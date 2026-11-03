# Phase 4 UI redesign — patient-friendly MRI study view

This is an **additive/overwrite patch** for the existing project. Extract the ZIP at the project root (the folder that contains `app.py`, `demo/`, `notebooks/`, and `src/`). It does not replace any Phase 1–3 inference, preprocessing, model, or checkpoint files.

## What changes in the dashboard

1. **Patient information** is read from non-identifying DICOM tags only: age, sex, laterality, study date, body part, and modality. Names and patient IDs are never read or shown. A value missing in the sample is shown as `Không có dữ liệu`; the UI does not infer an age or sex.
2. **Exactly three MRI galleries** are rendered: Sagittal (mặt phẳng dọc), Coronal (mặt phẳng trán), and Axial (mặt phẳng ngang). All selected series matching a plane are grouped into that plane's gallery. Each page shows 12 thumbnails; use the page slider to browse further slices. The preview uses display-only windowing and thumbnail resizing.
3. **Doctor/radiology report** is displayed as readable text rather than a JSON object. The original text and language are preserved; only whitespace and line breaks are normalized. A header containing `StudyInstanceUID` is checked before display.
4. **Twelve model outputs** are presented with Vietnamese names, plain-language descriptions, and percentage bars. The output is still a model score, not a clinically calibrated probability or diagnosis.
5. Raw JSON provenance is no longer printed on the main page. Developer-only provenance is summarized as labelled text in a collapsed expander.

## Important: the current Phase 1 sample does not contain the report

The existing Phase 1 exporter included DICOM, `series_metadata.csv`, `series_selection.csv`, and `manifest.json`, but did not export the `Report` column. The RSNA competition's `train.csv` contains the free-text report for training studies; the report is not provided in `test.csv` at scoring time. See the official competition data page: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data.

This patch therefore adds:

```text
notebooks/15_export_demo_radiology_report.ipynb
```

Run that notebook in Kaggle with the RSNA competition data attached. Its default `SAMPLE_STUDY_UID` matches the Study UID in the currently shared local sample path. If your current `demo_sample/manifest.json` contains another UID, update the notebook variable to that exact value. The notebook reads one row from `train.csv`, preserves the `Report` text, and creates `demo_radiology_report.zip` containing `radiology_report.txt`. It does not copy DICOM, export the 12 target labels, or run inference.

After downloading that ZIP:

1. Extract `radiology_report.txt`.
2. Copy it beside `manifest.json` in the current local sample folder, e.g. `C:\\rsna_demo\\demo_sample\\radiology_report.txt`.
3. Restart Streamlit. The dashboard finds the file automatically and verifies the report's Study UID.

The report can be multilingual. The dashboard does not automatically translate or reinterpret it, to avoid changing clinical meaning. It is labelled as the original source report.

## Patient information availability

The UI reads only these DICOM attributes when present: `PatientAge`, `PatientSex`, `Laterality` / `ImageLaterality`, `StudyDate`, `BodyPartExamined`, and `Modality`. It deliberately avoids `PatientName` and `PatientID`. If the DICOM sample does not contain a field, the UI shows `Không có dữ liệu`.

## Integration layout

```text
project-root/
├── app.py                                  # existing Phase 4 entry point, unchanged
├── requirements-ui.txt                    # updated: adds Pillow for thumbnails
├── notebooks/
│   └── 15_export_demo_radiology_report.ipynb  # new Kaggle helper notebook
├── src/rsna_knee/ui/
│   ├── streamlit_app.py                    # replaced: patient-friendly three-gallery UI
│   ├── dicom_display.py                    # replaced: adds thumbnails
│   ├── study_context.py                    # new: patient/report formatting and validation
│   └── dashboard_data.py                   # existing Phase 4 file, unchanged
└── tests/
    ├── test_dicom_display.py                # updated
    └── test_study_context.py                # new
```

## Install, test, and run

From the project root, with the existing virtual environment activated:

```powershell
python -m pip install -r requirements-ui.txt
python -m pytest tests/test_dashboard_data.py tests/test_dicom_display.py tests/test_study_context.py -q
streamlit run app.py
```

The UI reads existing Phase 3 ensemble artifacts and existing Phase 1 DICOM. It does not load DINOv2/ViT checkpoints and does not rerun either model.

## Interpretation boundary

This is a software demonstration using a single `gold_validation` sample. It is not a diagnostic device, the scores are not calibrated clinical risks, and neither the source radiology report nor the model outputs should be treated as a substitute for a radiologist's assessment.
