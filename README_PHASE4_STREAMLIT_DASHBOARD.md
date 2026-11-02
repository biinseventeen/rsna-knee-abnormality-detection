# Phase 4 — Streamlit dashboard (project-root patch)

## Integration rule

This is an **additive patch for the existing RSNA project**. The ZIP has no enclosing `rsna_phase4/` directory. Extract/merge its contents directly into the existing project root — the directory that already contains `configs/`, `demo/`, `notebooks/`, `scripts/`, `src/`, and `pyproject.toml`.

The final layout is:

```text
project-root/
├── app.py                                  # new: Streamlit entry point
├── requirements-ui.txt                    # new: optional UI dependencies
├── README_PHASE4_STREAMLIT_DASHBOARD.md    # new: this guide
├── demo/
│   ├── input/                              # existing Phase 1 sample
│   └── output/ensemble/                   # existing Phase 3 outputs
├── src/rsna_knee/
│   ├── dicom/                              # existing Phase 1 code
│   ├── inference/                          # existing Phase 2–3 code
│   ├── models/                             # existing
│   ├── preprocessing/                      # existing
│   ├── report/                             # existing
│   ├── vision/                             # existing
│   └── ui/                                 # new: dashboard package
│       ├── __init__.py
│       ├── dashboard_data.py
│       ├── dicom_display.py
│       └── streamlit_app.py
└── tests/
    ├── test_dashboard_data.py              # new
    └── test_dicom_display.py               # new
```

No existing Kaggle notebook, checkpoint, model, preprocessing module, or Phase 1–3 source file is replaced. The only existing directory extended is `src/rsna_knee/`, by adding the new `ui/` subpackage.

## What the dashboard does

- Shows the DINOv2, ViT-B/16, and final ensemble probabilities for the canonical 12 targets.
- Shows the Phase 3 blend/fallback status and model provenance.
- Compares the two model outputs and the final ensemble in a grouped chart and table.
- Loads the Phase 1 DICOM sample, allows series/slice selection, and renders a display-only MRI preview.
- Verifies that the DICOM `StudyInstanceUID` matches the prediction artifacts before displaying MRI images.

**It is read-only with respect to model inference:** it does not load checkpoints, initialize DINOv2/ViT, or run a forward pass.

## Dependencies and run command

Activate the existing project `.venv` first. From the project root:

```powershell
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

The base project's `requirements.txt` / `pyproject.toml` are not modified. The UI requirements list the dependencies needed by this dashboard without installing PyTorch or model-training packages again.

## Default paths

The sidebar starts with:

- Phase 3 output: `demo/output/ensemble`
- Extracted Phase 1 DICOM sample: prefers `C:\rsna_demo\demo_sample` when present; otherwise uses `demo/input/demo_sample`.

**Windows path-length note:** the full DICOM path includes the project root, `dicom/`, a Study UID folder, a Series UID folder, and a SOP UID filename. This can exceed the traditional 260-character `MAX_PATH` limit. The sample previously audited at `C:\rsna_demo\demo_sample` is intentionally preferred because its shorter root keeps the full paths shorter. You can still edit the path in the sidebar. The sample loader also accepts a Phase 1 sample ZIP, or a parent directory containing exactly one exported sample package.

The sample directory must contain `manifest.json`, `series_metadata.csv`, `series_selection.csv`, and `dicom/` (the Phase 1 package layout).

## Phase 3 data contract checked at dashboard startup

The dashboard requires **all four** files in `demo/output/ensemble/`:

```text
ensemble_prediction.csv
ensemble_prediction_long.csv
ensemble_prediction.npz
ensemble_report.json
```

Before displaying predictions, it checks:

1. One common `StudyInstanceUID` across the wide CSV, long CSV, NPZ, and report.
2. Canonical 12-target order in the CSV, NPZ, and report.
3. Finite probabilities in `[0, 1]` and equality of model/ensemble vectors across CSV and NPZ representations.
4. Model validity across the long CSV, NPZ, and JSON report.
5. Configured/effective weights across all artifacts.
6. The weighted 90/10 formula or the correct DINOv2-only, ViT-only, or neutral-0.5 fallback policy.

If any required artifact is missing or inconsistent, the dashboard refuses to show the predictions as valid.

## Tests

From project root:

```powershell
python -m pytest tests/test_dashboard_data.py tests/test_dicom_display.py -q
```

The tests cover all four Phase 3 output contracts, all three fallback states, probability and identity mismatch rejection, plus DICOM display conversion. These tests do not replace launching the dashboard with the actual local DICOM sample.

## MRI display note

The viewer uses DICOM `WindowCenter` / `WindowWidth` when present, otherwise a 1st–99th percentile display window. It applies rescale slope/intercept and inverts `MONOCHROME1` images. This is a **visualization-only transform**; it is not the Kaggle-matched preprocessing used by DINOv2 or ViT and must not be reused as model input preprocessing.

## Interpretation boundary

The probabilities belong to one sample exported from `gold_validation`. They are model outputs for a software demonstration, not clinical diagnoses, not calibrated disease probabilities, and not a new independent validation estimate or Kaggle score.
