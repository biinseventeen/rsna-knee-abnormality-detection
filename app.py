"""Entry point for the local RSNA Knee Streamlit dashboard.

Run from the repository root:
    streamlit run app.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if not SRC_ROOT.is_dir():
    raise RuntimeError(f"Không tìm thấy source directory: {SRC_ROOT}")
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rsna_knee.ui.streamlit_app import main  # noqa: E402


if __name__ == "__main__":
    main()
