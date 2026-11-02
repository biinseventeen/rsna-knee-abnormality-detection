# Phase 2E — Full-study ViT-B/16 inference

## Goal

Run the final Residual Attention Pooling ViT-B/16 on one complete preprocessed study from Phase 2D. The network receives the fixed 36-slice tensor and its complete metadata/mask arrays, encodes all valid slices, performs study aggregation, and returns 12 sigmoid probabilities.

## Source of truth

- Final Kaggle notebook: `06b_vit_residual_attention_submission_kaggle.ipynb`, model inference section.
- Exact model call:

  ```python
  logits = model(images, valid, plane_ids, fluid, fat, positions)
  probs = torch.sigmoid(logits.float())
  ```

- Local architecture/checkpoint loader: Phase 2D `vit_study.py` and `vit_checkpoint.py`.
- Input preprocessing cache: `demo/output/vit_preprocessing/vit_preprocessed_sample.npz` from Notebook 11.

The model's `forward()` applies `/255`, grayscale-to-RGB replication, ImageNet normalization, valid-slice selection and internal slice chunking. Do not normalize the Phase 2D uint8 images again.

## Files added

```text
src/rsna_knee/inference/
├── __init__.py
└── vit_full_study.py

notebooks/
└── 13_local_vit_full_study_inference.ipynb

tests/
└── test_vit_full_study_inference.py
```

## Requirements

Phase 2D must already be integrated and passed on the local machine:

- `src/rsna_knee/preprocessing/vit.py`
- `src/rsna_knee/models/vit_study.py`
- `src/rsna_knee/models/vit_checkpoint.py`
- `demo/output/vit_preprocessing/vit_preprocessed_sample.npz`

The final checkpoint artifact should be discoverable under `artifacts/`, preferably:

`artifacts/checkpoints/102526/vit_pretrained_centerprior_model_artifacts.zip`

If it is elsewhere, update `VIT_ARTIFACT_SOURCE` in Notebook 13 or adjust artifact resolution.

## Run

1. Extract this patch into the existing project root, preserving paths.
2. Open `notebooks/13_local_vit_full_study_inference.ipynb` in VS Code.
3. Run all cells.
4. The notebook loads the Phase 2D cache, resolves the final artifact, restores the model with `strict=True`, runs one full-study forward pass, and writes outputs.

CPU uses `RUNTIME_SLICE_CHUNK_SIZE = 2`; reduce to 1 in Notebook 13 if RAM is constrained. This is runtime chunking only, not a change to model weights or the input contract. The implementation deliberately does not enable AMP, matching the final Kaggle test inference call.

## Outputs

```text
demo/output/vit_full_inference/
├── vit_full_study_predictions.csv
├── vit_full_study_prediction.npz
└── vit_full_study_inference_report.json
```

- CSV: target, raw logit, probability.
- NPZ: study UID, ordered targets, logits and probabilities.
- JSON: checkpoint provenance, strict-load state, valid slice counts by plane, tensor/output shapes and timing.

No neutral 0.5 fallback is emitted for a local study with zero valid slices. That condition raises an explicit error rather than pretending that a model prediction was produced. Competition submission fallback policy remains separate and unchanged.

## Automated validation

The tests use a deterministic fake model to validate the input contract, forward argument order, full valid-mask propagation, probability conversion, output persistence, and invalid-input handling. They do not load the user's real ViT checkpoint. The real checkpoint and real DICOM study are validated when Notebook 13 is run locally.

## Completion criteria

- [ ] Phase 2D NPZ contract passes.
- [ ] Final ViT artifact is found.
- [ ] Checkpoint restores with `strict=True`.
- [ ] All valid slices in the 36-slot study are passed to the model.
- [ ] The forward call returns 12 finite logits and 12 probabilities in `[0, 1]`.
- [ ] CSV, NPZ and JSON report are saved.
- [ ] Notebook prints `PHASE 2E: ALL CRITERIA PASSED`.

This is software demonstration inference on the selected `gold_validation` sample, not an independent validation estimate or clinical diagnosis.
