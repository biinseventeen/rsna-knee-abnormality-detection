# Phase 3 — Local 90/10 ensemble integration

## Goal

Combine the already-computed one-study outputs of:

- Phase 2C: five-fold DINOv2 ViT-S/14 five-role 2.5D ensemble.
- Phase 2E: full-study ViT-B/16 Residual Attention Pooling.

This phase does **not** reload checkpoints or rerun either model. It validates each saved output and computes the final weighted probabilities.

## Exact blend policy

The implementation matches the final Kaggle ensemble policy:

- Both models valid: `0.90 * DINOv2 + 0.10 * ViT-B/16`.
- Only DINOv2 valid: use DINOv2 probabilities unchanged.
- Only ViT valid: use ViT probabilities unchanged.
- Neither valid: emit neutral `0.5` for all 12 targets and mark the report `fallback`.

A missing output file or malformed report is treated as an integration error, not silently as an invalid model. This prevents a crashed inference job from being mistaken for a legitimate per-study modality fallback.

## Inputs

The default local output folders are:

```text
demo/output/dinov2_full_inference/
├── dinov2_five_fold_predictions.npz
└── dinov2_five_fold_inference_report.json

demo/output/vit_full_inference/
├── vit_full_study_prediction.npz
└── vit_full_study_inference_report.json
```

The DINOv2 NPZ contains `study_uid`, `targets`, `status`, `fold_probabilities` and `ensemble_probabilities`.
The ViT NPZ contains `study_uid`, `targets`, `logits` and `probabilities`.

## Validation

Before blending, the module verifies:

- Both artifacts have the same `StudyInstanceUID`.
- Target order matches the canonical 12 competition labels.
- Each probability vector has shape `[12]`, finite values and range `[0, 1]`.
- DINOv2 five-fold probabilities have shape `[5, 12]` and the saved mean equals their arithmetic mean.
- DINOv2 NPZ and JSON report statuses agree.
- A ViT result counts as valid only when the report says `success`, strict checkpoint loading is true, and at least one slice was used.

## Outputs

Saved under `demo/output/ensemble/`:

- `ensemble_prediction.csv`: one-row competition-shaped file with `StudyInstanceUID` and 12 target columns.
- `ensemble_prediction_long.csv`: per-target DINOv2, ViT and final probabilities, validity and effective weights.
- `ensemble_prediction.npz`: numeric vectors, targets, study ID, configured/effective weights and blend status.
- `ensemble_report.json`: source provenance, validity, blend policy, effective weights and output summary.

## Run

1. Apply this patch into the existing project root; it only adds the ensemble module, notebook, tests and this README.
2. Ensure Phase 2C and Phase 2E completed successfully and their output files exist.
3. Open `notebooks/14_local_ensemble_integration.ipynb` and run all cells.
4. If outputs are stored in different locations, update the two source paths in the notebook.

Default weights are `DINOv2=0.90`, `ViT-B/16=0.10`.

## Scope note

The selected sample is from the `gold_validation` split. The blended result is a software demonstration only; it is not an independent validation estimate, a new competition score, or a clinical diagnosis.
