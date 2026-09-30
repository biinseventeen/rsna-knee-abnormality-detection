from __future__ import annotations

import argparse
from pathlib import Path

from .config import LABELS
from .dicom.io import find_dicom_files, load_dicom_header


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsna-knee",
        description="RSNA Knee Abnormality Detection CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect-dicom", help="Inspect DICOM inputs.")
    inspect.add_argument("inputs", nargs="+", type=Path)

    predict = sub.add_parser(
        "predict",
        help="Run DICOM -> DINOv2 -> student -> 12-label prediction.",
    )
    predict.add_argument("inputs", nargs="+", type=Path)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--output", type=Path, default=Path("prediction.json"))
    predict.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    return parser


def _run_inspect(inputs: list[Path]) -> int:
    files = find_dicom_files(inputs)
    print(f"Found {len(files)} DICOM files.")
    if not files:
        return 1

    for path in files[:20]:
        h = load_dicom_header(path)
        print(
            f"- {path.name}: "
            f"Study={h.get('StudyInstanceUID', '?')} "
            f"Series={h.get('SeriesInstanceUID', '?')} "
            f"Instance={h.get('InstanceNumber', '?')} "
            f"Shape={h.get('Rows', '?')}x{h.get('Columns', '?')} "
            f"Modality={h.get('Modality', '?')}"
        )

    if len(files) > 20:
        print(f"... and {len(files) - 20} more files.")
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train/export the student checkpoint before running the MVP."
        )

    from .inference.predict import predict_from_paths, save_prediction

    prediction = predict_from_paths(
        inputs=args.inputs,
        checkpoint=args.checkpoint,
        device=args.device,
        labels=LABELS,
    )
    save_prediction(prediction, args.output)
    print(f"Prediction saved to: {args.output}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-dicom":
        return _run_inspect(args.inputs)
    if args.command == "predict":
        return _run_predict(args)

    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
