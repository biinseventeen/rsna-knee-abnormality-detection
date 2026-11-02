# Phase 2C — Full five-fold DINOv2 inference (one local study)

## Goal

Run the final five-fold DINOv2 model on the real one-study preprocessing cache from Phase 2A.
Each fold is restored with the Phase 2B strict checkpoint loader, runs a full-study forward pass,
produces 12 sigmoid probabilities, and is released before the next fold is loaded. The final
prediction is the arithmetic mean of the five per-fold probability vectors.

This follows the final Kaggle ensemble notebook's inference contract:

- Model: `FiveRoleDINOv2_2p5D` (Kaggle Cell 29).
- Checkpoint restore: bundled DINOv2 ViT-S/14 source, `pretrained=False`, `strict=True`
  (Kaggle Cell 31; Phase 2B loader).
- Study forward: `[1, 5, 10, 3, 224, 224]` images plus `[1, 5, 10]` validity/position arrays.
- Each fold: `sigmoid(logits.float())`.
- Five-fold aggregation: `fold_probabilities.mean(axis=0)` (Kaggle Cell 33).

## Why models are loaded one at a time

The Kaggle notebook keeps fold models available in its GPU inference context. The local demo may
run on CPU and has tighter memory constraints. This implementation restores one fold, predicts,
keeps only its 12 probabilities and metadata, then deletes the model, collects garbage and clears
CUDA cache when applicable. It does not change the fold aggregation rule.

Defaults:

- CUDA available: encoder chunk 8; AMP enabled by default.
- CPU: encoder chunk 2; AMP disabled automatically.
- Set `ENCODER_CHUNK_SIZE = 1` in the notebook if RAM is constrained. This changes only execution
  chunking, not learned weights or the model's mathematical input contract.

## Files added

```text
src/rsna_knee/inference/
├── __init__.py
└── dinov2_ensemble.py

notebooks/
└── 10_local_dinov2_full_inference.ipynb

tests/
└── test_dinov2_ensemble_inference.py
```

These files use the model and bundle modules added in Phase 2B, and the NPZ from Phase 2A.
Do not replace the Kaggle competition notebook or its submission artifacts.

## Inputs required

1. Phase 2A cache:

```text
demo/output/dinov2_preprocessing/
├── dinov2_preprocessed_sample.npz
└── dinov2_preprocessing_summary.json
```

2. Final extracted five-fold DINOv2 bundle or the full bundle ZIP. It must contain:

```text
bundle_manifest.json
dinov2_repo/hubconf.py
dinov2_repo/dinov2/
five_role_2p5d_v2_cv5/fold_1/best_checkpoint.pt
...
five_role_2p5d_v2_cv5/fold_5/best_checkpoint.pt
```

The generic/older `artifacts/teacher/dinov2-artifact.zip` is not assumed to be this final bundle.

## Run

1. Extract this patch into the project root, preserving folder structure.
2. Make sure Phase 2A and Phase 2B files are already integrated.
3. Open `notebooks/10_local_dinov2_full_inference.ipynb` in VS Code.
4. Run all cells. Set `DINO_BUNDLE_SOURCE` if automatic discovery cannot locate the final bundle.
5. If the machine has limited RAM, set `ENCODER_CHUNK_SIZE = 1` before starting full inference.

## Outputs

Saved under `demo/output/dinov2_full_inference/`:

- `dinov2_fold_probabilities.csv` — each fold's 12 probabilities.
- `dinov2_ensemble_prediction.csv` — five-fold mean for the study.
- `dinov2_five_fold_predictions.npz` — numeric arrays, targets, role order and study ID.
- `dinov2_five_fold_inference_report.json` — runtime, device, strict-load metadata, input validity,
  fold information and aggregation details.

No binary labels or medical diagnosis are generated; the outputs remain probabilities.

## Important fallback behavior

The Kaggle submission initializes every fold probability to `0.5`. A study with no usable window
keeps that neutral value without model inference. The local function preserves this behavior and
marks status `all_roles_unusable_fallback_0.5`; such a run is **not** counted as a successful model
inference. If a fold itself errors, the local function raises and does not silently replace that
fold with `0.5` or return a partial ensemble.

## Verification scope

Automated tests use a deterministic fake fold loader/model to verify input validation, sequential
fold calls, sigmoid-before-mean aggregation, neutral fallback, and output serialization. They do
not prove the actual five local checkpoints have been run; that is confirmed only when the user's
notebook completes all five strict loads and full-study forward passes.

## Interpretation note

The sample was exported from `gold_validation`. It is a software demonstration input, not a new
independent evaluation. It must not be used to claim a new validation score. Across-fold standard
deviation is descriptive only and is not a calibrated uncertainty estimate.
