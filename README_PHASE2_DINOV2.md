# Phase 2A — Local DINOv2 five-role preprocessing

This patch adds the local DINOv2 preprocessing port for the RSNA knee demo.
It is based on the final Kaggle ensemble notebook `08_model_ensemble_kaggle_dataset_ready.ipynb`,
sections 12–14 (role order/selection, per-series DICOM preprocessing, and 2.5D window building).

## What is included

- `src/rsna_knee/preprocessing/dinov2.py`
  - Reads the role-to-series mapping from the `DemoStudy` object loaded by Phase 1.
  - Reads and sorts DICOM headers using projected `ImagePositionPatient`, then `InstanceNumber`,
    then natural filename order — same thresholds and tie-breakers as the Kaggle notebook.
  - Decodes DICOM pixels, applies modality LUT (with slope/intercept fallback), and inverts MONOCHROME1.
  - Samples up to 12 evenly spaced slices per role, with remaining slices used as fallback when a
    selected slice fails to decode.
  - Applies per-series 1.0/99.5 percentile clipping, scales to uint8, and resizes to 224×224 with
    `cv2.INTER_AREA`.
  - Builds 10 overlapping 3-slice windows for each of the five roles.
  - Returns `images` as `uint8 [5, 10, 3, 224, 224]`, `window_valid` as `[5, 10]`, and normalized
    `window_positions` as `[5, 10]`.

- `notebooks/08_local_dinov2_preprocessing.ipynb`
  - Loads the Phase 1 sample locally, runs only DINOv2 preprocessing, validates shapes and saves
    the arrays and diagnostics under `demo/output/dinov2_preprocessing/`.

- `tests/test_dinov2_preprocessing.py`
  - Tests the five-role tensor layout, 2-slice validity rule, overlapping windows, and missing-role
    behavior without needing DICOM files.

## Apply to the existing project

Extract this ZIP into the project root. It only adds the new preprocessing package/files and does
not replace the Kaggle competition notebooks or Phase 1 DICOM loader.

Install the additional imaging dependency in the same Python environment used by VS Code:

```powershell
python -m pip install "opencv-python-headless>=4.9"
```

Phase 1 already uses `pydicom`, `numpy`, and `pandas`.

## Run

Open `notebooks/08_local_dinov2_preprocessing.ipynb` in VS Code and run all cells.
The notebook searches these sample locations in order:

1. `C:\\rsna_demo\\demo_sample`
2. `<project-root>\\demo\\input\\demo_sample`

Edit `SAMPLE_SOURCE` if your sample is elsewhere.

## Important model-input contract

The preprocessing module deliberately does **not** divide by 255 or apply ImageNet mean/std.
The final Kaggle `FiveRoleDINOv2_2p5D.forward()` performs those operations internally:

```python
selected = flat_images.index_select(0, valid_indices).float().div_(255.0)
selected = (selected - self.image_mean) / self.image_std
```

Do not normalize the returned uint8 tensor a second time.

## Scope / current limitations

This phase ports DINOv2 preprocessing only. It does not load DINOv2 checkpoints, instantiate the
backbone, run model inference, or blend with ViT. Those are later phases.

The code's preprocessing logic has been ported against the notebook source and syntax/tests have
been checked in the build environment. The actual 164-file local DICOM sample is not mounted in
this build environment, so the full pixel-decoding notebook must still be run on the user's machine.
