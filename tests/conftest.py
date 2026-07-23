import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_DIR = ROOT / "streamlit_app"
for path in (ROOT, STREAMLIT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
