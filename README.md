# RSNA Knee Abnormality Detection

Weakly supervised MRI study classification for 12 knee abnormalities from the RSNA Knee MRI dataset. The repository contains research notebooks, model-specific preprocessing and inference code, and a Streamlit viewer for exported DICOM studies and saved predictions.

> **Scope:** This is a research/software demonstration, not a clinical diagnostic tool. Its scores are not clinically calibrated probabilities. The bundled demo sample is from `gold_validation` and does not represent independent validation.

## What to use

- **View the prepared demo:** run the Streamlit dashboard (instructions below). It reads saved Phase 3 predictions and a local DICOM sample; it does not run model inference.
- **Reproduce a local prediction:** run the local notebooks in order. You need the exported sample, both final trained model artifacts, and a Python environment with the required dependencies.
- **Explore training/research:** use the Kaggle-oriented notebooks with the RSNA competition data attached. The competition dataset and most model artifacts are not source-controlled.

## Quick start: dashboard

Use Python 3.10 or newer. From the repository root, install the package and optional dashboard dependencies, then start Streamlit:

```powershell
python -m pip install -e .
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

The dashboard defaults to `demo/output/ensemble` for saved predictions and scans the directory selected by `RSNA_DEMO_SAMPLE` (if set), otherwise `demo/input/demo_sample`, `demo/input`, or the legacy `C:\rsna_demo\demo_sample` folder. Select a study from the **Chọn study** dropdown in the sidebar. The package folder path is under **Thư viện study**; prediction and optional report paths are under **Cài đặt nâng cao**.

The dashboard requires these four Phase 3 files in its selected output directory:

```text
ensemble_prediction.csv
ensemble_prediction_long.csv
ensemble_prediction.npz
ensemble_report.json
```

The sample directory must contain `manifest.json`, `series_metadata.csv`, `series_selection.csv`, and `dicom/`. A `radiology_report.txt` can be added to show the source report; Notebook 15 exports it from Kaggle `train.csv`. The app checks that the sample and prediction artifacts refer to the same study before showing MRI images.

To browse multiple studies, place one complete exported package per study beneath a common directory, for example `demo/input/demo_sample/study_001/manifest.json` plus that package's `series_metadata.csv`, `series_selection.csv`, and `dicom/` tree. The dashboard scans `demo/input/demo_sample` and lists its study folders. Choose a study by its folder name and displayed Study UID; after copying in more packages, click **Làm mới danh sách study**. For scores, place each study's four Phase 3 files under `demo/output/ensemble/<StudyInstanceUID>/`. The dashboard matches the selected study to its output report. If there is no matching output yet, it still shows that study's metadata and MRI galleries.

## Reproduce a local prediction

### Inputs and dependencies

The local pipeline needs:

1. A Phase 1 `demo_sample` package (or its ZIP), with DICOMs and the metadata/selection files listed above.
2. The final five-fold DINOv2 bundle: `bundle_manifest.json`, the bundled local `dinov2_repo/` source, and `five_role_2p5d_v2_cv5/fold_1` through `fold_5/best_checkpoint.pt`.
3. The final ViT checkpoint `vit_pretrained_best_checkpoint.pt`, either directly or in `vit_pretrained_centerprior_model_artifacts.zip`.
4. PyTorch installed for the desired CPU/CUDA setup. Model artifacts are large and are not installed by `pip install -e .`.

Install the package, OpenCV used by model-specific preprocessing, and notebook/test dependencies as needed:

```powershell
python -m pip install -e .
python -m pip install -r requirements-phase2.txt
python -m pip install -r requirements-phase2b.txt
```

For the optional dashboard, also install `requirements-ui.txt`. The project declares PyTorch and Transformers in `pyproject.toml`; select a PyTorch build compatible with your machine if the default package resolver is unsuitable. The local DINOv2 bundle loader uses its bundled source and does not download DINOv2 weights. The older CLI path uses Hugging Face Transformers and may need a cached/downloaded model.

### Notebook order

From the repository root, open these notebooks in VS Code/Jupyter and run all cells in order:

1. `notebooks/06_export_demo_sample.ipynb` — export the one-study DICOM package from Kaggle (skip if you already have it).
2. `notebooks/07_local_dicom_audit.ipynb` — inspect package structure and DICOM headers.
3. `notebooks/08_local_dinov2_preprocessing.ipynb` — build DINOv2 input arrays.
4. `notebooks/09_local_dinov2_model_restore.ipynb` — strictly restore a fold and smoke-test the model.
5. `notebooks/10_local_dinov2_full_inference.ipynb` — run the five-fold DINOv2 inference.
6. `notebooks/11_local_vit_preprocessing.ipynb` — build ViT study inputs.
7. `notebooks/12_local_vit_model_restore.ipynb` — strictly restore and smoke-test ViT.
8. `notebooks/13_local_vit_full_study_inference.ipynb` — run ViT inference.
9. `notebooks/14_local_ensemble_integration.ipynb` — validate and combine the saved predictions.
10. Start the dashboard with `streamlit run app.py`.

Set the bundle/artifact/sample paths in the notebook configuration cells if automatic discovery does not find your files. On CPU or memory-constrained machines, reduce the DINO encoder chunk size and ViT slice chunk size in their respective notebooks. Chunking controls runtime memory; it does not change the model inputs or weights.

To export the optional radiology report, run `notebooks/15_export_demo_radiology_report.ipynb` in Kaggle with the RSNA data attached, then place the resulting `radiology_report.txt` beside the sample `manifest.json`. This notebook reads a report from `train.csv`; reports are not present in competition `test.csv`.

## What the pipeline does

```text
DICOM sample + series metadata/selection
  -> sample validation and ordered DICOM headers
  -> DINOv2 preprocessing: five roles, 10 three-slice windows per role
  -> ViT preprocessing: up to 36 slices with plane and sequence metadata
  -> separate model inference (five DINOv2 folds; one ViT study model)
  -> validated 90% DINOv2 / 10% ViT blend
  -> CSV, NPZ and JSON artifacts
  -> read-only Streamlit dashboard
```

The two preprocessing branches are intentionally distinct. DINOv2 expects `uint8 [5, 10, 3, 224, 224]`, validity masks and window positions. ViT expects a fixed 36-slice study plus validity, plane, fluid, fat-suppression and position arrays. Do not pass generic `dicom.preprocess` output to these trained models.

When both model outputs are valid, the blend is `0.90 * DINOv2 + 0.10 * ViT`. If one model is invalid, the other is used alone. If neither is valid, the ensemble records an explicit neutral 0.5 fallback. Missing or malformed prediction files are treated as errors. The dashboard independently checks study identity, target ordering, ranges, and consistency between the saved CSV, NPZ and report.

## Other commands

Audit a local sample from the command line:

```powershell
python scripts/audit_demo_sample.py demo/input/demo_sample
```

The package also exposes `rsna-knee`:

```powershell
rsna-knee --help
rsna-knee inspect-dicom demo/input
rsna-knee predict demo/input --checkpoint artifacts/checkpoints/student.pt --output demo/prediction.json
```

**Important:** `rsna-knee predict` is an older/simple center-triplet DINOv2 feature + multilabel-head path (`inference/predict.py` and `vision/`). It is not the five-role DINOv2 + ViT ensemble described above, and it requires a checkpoint compatible with `vision.model.MultiLabelHead`.

## Tests

Install pytest (`requirements-phase2b.txt` includes it), then run from the repository root:

```powershell
python -m pytest
```

For the dashboard-specific data, DICOM display, and study-context tests:

```powershell
python -m pytest tests/test_dashboard_data.py tests/test_dicom_display.py tests/test_study_context.py -q
```

The tests use synthetic inputs and fake models in many cases; they do not prove that your real DICOM sample or local model checkpoints run successfully.

## Repository map

| Path | Contents |
|---|---|
| `src/rsna_knee/dicom/` | DICOM reading, sample package validation, audit and basic utilities |
| `src/rsna_knee/preprocessing/` | DINOv2 and ViT-specific preprocessing |
| `src/rsna_knee/models/` | DINOv2/ViT architectures and strict checkpoint loaders |
| `src/rsna_knee/inference/` | Per-model inference, artifact validation and ensemble blending |
| `src/rsna_knee/ui/` | Streamlit app, prediction validation, DICOM thumbnail and report display |
| `src/rsna_knee/vision/` | Older CLI backbone and multilabel head |
| `src/rsna_knee/report/` | Report label parsing and teacher evaluation helpers |
| `notebooks/` | Kaggle research/training and local demo workflows |
| `tests/` | Unit tests for the package contracts |
| `configs/` | Teacher and student workflow configuration |
| `artifacts/`, `demo/output/` | Model artifacts and generated output locations; most generated content is gitignored |

## Notebook guide

- `01_eda.ipynb`, `02_teacher_eval.ipynb`, `03_a_generate_pseudo_labels.ipynb`, `03_b_generate_pseudo_labels_missing.ipynb`: data exploration and report-derived labels.
- `4a`–`4g` notebooks: experimental LSTM, ViT and DINOv2 training/inference/ensemble workflows.
- `05_demo.ipynb`: small demo notebook.
- `06`–`15` notebooks: local sample export, audit, model-specific preprocessing/restoration/inference, ensemble integration, and optional report export.

The training/research notebooks may require the RSNA competition data, Kaggle environment, GPU memory, or external teacher-model access. They are not required to view an already-prepared dashboard output.

## Data and output locations

Competition data is not part of the source package. `demo/input/`, `demo/output/`, and most checkpoint/feature directories are gitignored; keep your local input and generated artifacts backed up separately. The sample audit command writes `demo/output/sample_audit.json`. Local model stages write caches and reports beneath `demo/output/`; Phase 3 writes the four ensemble files listed above.

## How this repository evolved

The local demo was added in stages: DICOM audit (Phase 1), DINOv2 preprocessing/model/inference (Phases 2A–2C), ViT preprocessing/model/inference (Phases 2D–2E), ensemble integration (Phase 3), and Streamlit UI (Phase 4). The old phase-specific README files have been consolidated here; consult notebook source and tests for implementation-level contracts.
