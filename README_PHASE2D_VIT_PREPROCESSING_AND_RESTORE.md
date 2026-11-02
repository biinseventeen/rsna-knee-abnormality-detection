# Phase 2D — ViT-B/16 preprocessing + architecture/checkpoint restoration

## Scope

This patch adds the local ViT-B/16 branch for the RSNA knee demo. It does not edit any Kaggle competition notebook. The source of truth is the final ensemble notebook:

- Cell 7: one series selection per anatomical plane (the local sample already carries the selected plane-to-series mapping in `series_selection.csv`).
- Cell 9: DICOM sort, pixel decoding, per-series percentile normalization, 12-slice sampling, 224x224 resize, and 36-slice study packing.
- Cell 17: `PretrainedSliceEncoder`, `StudyFeatureAggregator`, `PretrainedViTStudyModel`, run-config parsing, and strict checkpoint loading.

## Added files

```text
src/rsna_knee/preprocessing/vit.py
src/rsna_knee/models/vit_study.py
src/rsna_knee/models/vit_checkpoint.py
notebooks/11_local_vit_preprocessing.ipynb
notebooks/12_local_vit_model_restore.ipynb
tests/test_vit_preprocessing.py
tests/test_vit_model.py
```

## Preprocessing contract

Input is the local `DemoStudy` object returned by Phase 1. The module uses `study.series_for_vit_plane()` for the exported selection of `Sagittal`, `Coronal`, and `Axial` series. It does not re-run series selection on the local sample.

It mirrors final Kaggle ViT Cell 9:

- Sort `.dcm` files by projected `ImagePositionPatient` when at least 70% have usable positions; otherwise use `InstanceNumber` with the same threshold; otherwise natural filename order.
- Sample up to 12 evenly spaced indices first, then use remaining indices as decode fallbacks until 12 slices have decoded.
- Decode `pixel_array`, handle 2D/3D arrays, apply modality LUT with slope/intercept fallback, invert MONOCHROME1, and zero non-finite values.
- Normalize each series independently using percentile range `[1.0, 99.5]`, map to uint8, resize to 224x224 with `cv2.INTER_AREA`.
- Concatenate slices in `Sagittal`, `Coronal`, `Axial` order, maximum 12 slices per plane and 36 total.
- Attach plane ID, fluid flag, fat-suppression flag and normalized sorted-slice position to each valid slice.

Output arrays:

- `images`: `uint8 [36, 224, 224]`
- `valid`: `uint8 [36]`
- `plane_ids`: `int8 [36]` (`0=sagittal`, `1=coronal`, `2=axial`; padding `-1`)
- `fluid`, `fat`: `int8 [36]` (`-1` means unknown)
- `positions`: `float32 [36]`

No ImageNet normalization is applied in preprocessing. `PretrainedViTStudyModel.forward()` applies `/255`, grayscale-to-RGB replication and ImageNet normalization internally. Do not normalize the saved uint8 images again.

## Model/checkpoint contract

The model source follows final Kaggle Cell 17. It instantiates torchvision `vit_b_16(weights=None)` because the fine-tuned checkpoint includes the entire trained model state. It validates checkpoint target order and the saved `run_config`, then loads `checkpoint["model_state"]` with `strict=True`.

The runtime `slice_chunk_size` may be reduced on CPU (for example, 1 or 2) to bound inference memory. It changes only how valid slices are chunked through the encoder; it is not a learned parameter and does not change the state-dict structure.

Accepted artifact inputs:

- `vit_pretrained_best_checkpoint.pt`
- `vit_pretrained_centerprior_model_artifacts.zip` containing exactly one `vit_pretrained_best_checkpoint.pt`

Default project location used by the notebook:

`artifacts/checkpoints/102526/vit_pretrained_centerprior_model_artifacts.zip`

Override `VIT_ARTIFACT_SOURCE` in notebook 12 if your artifact is elsewhere.

## Run order

1. Add this patch ZIP to the existing project root without deleting previous phase files.
2. Run `notebooks/11_local_vit_preprocessing.ipynb`.
3. Confirm its criteria and that `demo/output/vit_preprocessing/vit_preprocessed_sample.npz` exists.
4. Run `notebooks/12_local_vit_model_restore.ipynb`.
5. Confirm strict load and quick one-valid-slice smoke test.

## Important scope limitation

This phase stops after preprocessing and checkpoint restoration/smoke test. It does not yet run the full 36-slice ViT inference, compute final ViT predictions, or blend with DINOv2. Those will be the next phase after the local checks pass.
