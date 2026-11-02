# Phase 2B — Restore exact DINOv2 five-role model (one fold)

## Mục tiêu

Port đúng kiến trúc `FiveRoleDINOv2_2p5D` từ notebook Kaggle ensemble cuối và load **một fold** bằng `strict=True` trên máy local.

Nguồn chuẩn đã dùng:

- `08_model_ensemble_kaggle_dataset_ready.ipynb`, Cell 29: model architecture.
- `08_model_ensemble_kaggle_dataset_ready.ipynb`, Cell 31: local DINOv2 ViT-S/14 construction và cách load fold checkpoints.
- `bundle_manifest.json`: model dimensions, role order, target order và số fold.
- Phase 2A `dinov2_preprocessed_sample.npz`: input tensor contract.

Không sửa hoặc thay thế notebook Kaggle competition. Không training lại model.

## Những file được thêm

```text
src/rsna_knee/models/
├── __init__.py
├── dinov2_five_role.py   # Exact FiveRoleDINOv2_2p5D class
└── dinov2_bundle.py      # Bundle validation + local backbone + one-fold loader

notebooks/
└── 09_local_dinov2_model_restore.ipynb

tests/
├── conftest.py
├── test_dinov2_model.py
└── test_dinov2_bundle_validation.py
```

## Artifact bắt buộc

Notebook cần **final five-fold DINOv2 bundle** được tạo cho submission. Bundle phải chứa:

```text
bundle_manifest.json
dinov2_repo/
├── hubconf.py
└── dinov2/
five_role_2p5d_v2_cv5/
├── fold_1/best_checkpoint.pt
├── fold_2/best_checkpoint.pt
├── fold_3/best_checkpoint.pt
├── fold_4/best_checkpoint.pt
└── fold_5/best_checkpoint.pt
```

`artifacts/teacher/dinov2-artifact.zip` (file khoảng 8 MB trong danh sách artifacts cũ) không được xem là bundle cuối mặc định. Chỉ dùng nó nếu kiểm tra thực tế xác nhận đầy đủ manifest, source và năm checkpoint.

Có thể đặt thư mục đã giải nén ở ví dụ:

```text
artifacts/dinov2/five_role_v2_cv5_submission_bundle/
```

Hoặc đặt ZIP đầy đủ ở:

```text
artifacts/dinov2/five_role_v2_cv5_submission_bundle.zip
```

Nếu dùng vị trí khác, chỉnh biến `DINO_BUNDLE_SOURCE` trong notebook 09. Khi trỏ tới ZIP, notebook giải nén bundle vào `artifacts/_extracted_final_dinov2_bundle/`.

## Phase 2A output bắt buộc

```text
demo/output/dinov2_preprocessing/dinov2_preprocessed_sample.npz
```

NPZ phải có `images`, `window_valid`, `window_positions`, `role_order`, `role_series_uids`.

## Cài đặt

PyTorch phải được cài theo cấu hình phù hợp với máy. Nếu project chưa có torch, hãy cài bản tương thích với Python và CPU/GPU của máy trước. DINOv2 source được load trực tiếp từ bundle bằng `torch.hub.load(..., source="local", pretrained=False)`; không cần tải pretrained weights từ Internet.

## Chạy

Mở `notebooks/09_local_dinov2_model_restore.ipynb` trong VS Code và Run All.

Notebook sẽ:

1. Nạp Phase 2A NPZ và xác nhận shapes/role order.
2. Tìm/kiểm tra final five-fold bundle.
3. Đọc metadata của fold 1.
4. Khởi tạo DINOv2 ViT-S/14 từ bundled source với `pretrained=False`.
5. Dựng `FiveRoleDINOv2_2p5D` theo manifest.
6. Load `checkpoint["model_state"]` bằng `strict=True`.
7. Chạy quick forward smoke test với một valid window mỗi role (5 windows thay vì 50).
8. Lưu `demo/output/dinov2_model_restore/dinov2_fold1_restore_report.json`.

CUDA được chọn tự động nếu khả dụng; nếu không thì CPU. Loader chỉ giữ một fold trong memory. Encoder chunk mặc định là 8 trên CUDA, 2 trên CPU. `encoder_chunk_size` không phải tham số học được và không làm thay đổi state-dict keys.

## Compatibility checks đã thực hiện trong môi trường tạo patch

- So sánh class mới với nguyên class ở Kaggle Cell 29 bằng cùng dummy backbone: **94 state-dict keys theo cùng thứ tự**, không có missing/extra keys, không có shape mismatch.
- So sánh forward output của hai class trên cùng input/dummy backbone và cùng weights: max absolute difference **0.0**.
- Unit tests: state-dict strict round-trip, forward shape/probability bounds, reject study không có valid windows, bundle discovery và safe ZIP extraction.

**Giới hạn xác nhận:** trong môi trường tạo patch không có final five-fold checkpoint mà bạn đang giữ trên máy, nên chưa thể tuyên bố checkpoint thật đã load thành công. Bằng chứng quyết định của Phase 2B là notebook local của bạn chạy qua `model.load_state_dict(..., strict=True)` trên `fold_1/best_checkpoint.pt` và quick forward smoke test trả `[1, 12]` logits finite.

## Phase 2B completion criteria

- [ ] Notebook resolves đúng final bundle.
- [ ] Manifest target/role order trùng chính xác Kaggle.
- [ ] Fold 1 metadata hợp lệ.
- [ ] Strict state-dict load thành công, không missing/unexpected keys.
- [ ] Quick forward smoke test trả logits `[1, 12]`, finite.
- [ ] Restore report được lưu.

Phase này chưa load đủ 5 folds, chưa chạy full-study 50 windows, chưa averaging fold probabilities và chưa blend với ViT.
