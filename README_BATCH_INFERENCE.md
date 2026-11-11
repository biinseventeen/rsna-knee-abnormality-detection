# Batch inference for four additional RSNA demo studies — path fix

## Your current project layout

The notebook now automatically uses the directory shown in your VS Code screenshot:

```text
<project-root>/
└── demo/
    └── input/
        └── demo_sample/
            ├── study_001/
            │   ├── manifest.json
            │   ├── radiology_report.txt
            │   ├── series_metadata.csv
            │   ├── series_selection.csv
            │   └── dicom/
            ├── study_002/
            ├── study_003/
            ├── study_004/
            └── study_005/
```

No files need to be moved. `STUDIES_ROOT` is auto-detected as:
`PROJECT_ROOT / "demo" / "input" / "demo_sample"`.

It falls back to `C:\rsna_demo\studies` and `C:\rsna_demo\demo_sample` only if the project-local directory contains no valid exported packages.

## Run

1. Extract this ZIP into the project root, merging `notebooks/` and `scripts/`.
2. Open `notebooks/16_local_batch_inference_for_demo_studies.ipynb`.
3. Run cells from top to bottom.
4. In the preview table, confirm exactly four new studies are listed; `study_001` is excluded by StudyInstanceUID.
5. Confirm all four rows have `radiology_report=True`.

## Outputs

For each new study, the Phase 3 bundle is saved under:
`demo/output/ensemble/<StudyInstanceUID>/`

with the four files expected by the Phase 4 dashboard. Existing bundle for the first sample is not overwritten.
