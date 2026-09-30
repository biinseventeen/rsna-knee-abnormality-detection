# RSNA Knee Abnormality Detection

Weakly supervised MRI classification project.

## Pipeline

```text
Radiology reports
    -> LLM teacher
    -> pseudo / soft labels
    -> MRI studies
    -> DICOM preprocessing
    -> DINOv2-small features
    -> 12-label student head
    -> predictions / Kaggle submission

MVP:
A small folder of DICOM slices
    -> same preprocessing
    -> trained checkpoint
    -> 12 abnormality probabilities
```

## Local vs Kaggle

Local: code, EDA, report parser validation, DICOM smoke tests, and CPU-only development.
Kaggle: LLM inference, full DICOM feature extraction, GPU training, and competition inference.

The MRI competition dataset is not stored in this repository.

## CLI

```powershell
python -m pip install -e .
rsna-knee --help
rsna-knee inspect-dicom demo/input
rsna-knee predict demo/input --checkpoint artifacts/checkpoints/student.pt --output demo/prediction.json
```

`predict` is the final MVP interface. It requires a real trained student checkpoint.

## Notebooks

- `01_eda.ipynb`: dataset structure and label availability.
- `02_teacher_eval.ipynb`: evaluate report-to-label teacher on the 58 gold studies.
- `03_feature_extraction.ipynb`: DICOM -> DINOv2-small features.
- `04_student_training.ipynb`: train the 12-label student head.
- `05_demo_and_submission.ipynb`: final inference, submission, and demo smoke test.
