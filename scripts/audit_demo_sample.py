from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rsna_knee.dicom.sample import audit_demo_sample, load_demo_sample, write_audit_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an RSNA local demo DICOM sample exported from Kaggle."
    )
    parser.add_argument("source", type=Path, help="demo_sample.zip or extracted demo_sample directory")
    parser.add_argument("--extract-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("demo/output/sample_audit.json"))
    parser.add_argument("--skip-pixel-decode", action="store_true")
    args = parser.parse_args()

    study = load_demo_sample(args.source, extract_dir=args.extract_dir)
    report = audit_demo_sample(study, decode_pixels=not args.skip_pixel_decode)
    output = write_audit_report(report, args.output)

    print(f"StudyInstanceUID : {report['study_instance_uid']}")
    print(f"Series           : {report['series_count']}")
    print(f"DICOM files      : {report['dicom_file_count']}")
    print(f"DINO roles       : {report['dino_roles_selected']}/5")
    print(f"ViT planes       : {report['vit_planes_selected']}/3")
    print(f"Status           : {report['status']}")
    print(f"Warnings / errors: {report['warning_count']} / {report['error_count']}")
    print(f"Audit report     : {output}")
    if report["issues"]:
        print(json.dumps(report["issues"], ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
