# Local Demo — Phase 1: DICOM loading and audit

This patch adds a local-only DICOM sample loader and validator. It does **not** alter the Kaggle competition notebooks and does **not** run model inference yet.

## Expected sample archive

Use the `demo_sample.zip` produced by `06_export_demo_sample.ipynb`. Its expected contents are:

```text
demo_sample/
├── dicom/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
├── series_metadata.csv
├── series_selection.csv
├── manifest.json
└── README.txt
```

Place the ZIP at `demo/input/demo_sample.zip` in your existing project, or pass an extracted directory.

## Install/check dependencies

The original project requirements already include `pydicom`, `pandas`, and `numpy`:

```powershell
python -m pip install -e .
```

For a minimal standalone check:

```powershell
python -m pip install "pydicom>=3.0" "pandas>=2.2" "numpy>=1.26"
```

## Run the audit

From the project root:

```powershell
python scripts/audit_demo_sample.py demo/input/demo_sample.zip
```

Or against an extracted package:

```powershell
python scripts/audit_demo_sample.py demo/input/demo_sample
```

The ZIP is extracted to `demo/input/demo_sample_extracted/` by default. The command writes:

```text
demo/output/sample_audit.json
```

The audit checks archive/package structure, the manifest, metadata columns, the DINOv2 role and ViT plane mappings, the Study/Series UIDs embedded in DICOM headers, slice ordering metadata, and one representative pixel decode per selected series.

## Important boundary

The module only loads metadata and DICOM headers and runs an optional single-slice decode check. It intentionally does **not** normalize, resize, sample slices, construct ViT tensors, or construct DINOv2 2.5D windows. Those operations will be ported from the final Kaggle ensemble notebook as two separate preprocessing paths in the next phase. Do not substitute the old skeleton's `dicom_to_uint8()` implementation for either model path.
