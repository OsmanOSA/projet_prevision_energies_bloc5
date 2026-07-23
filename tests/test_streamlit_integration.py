"""Optional integration checks for the seven Streamlit pages.

Run locally with a PostgreSQL instance populated by the application:
    RUN_STREAMLIT_INTEGRATION=1 pytest -q tests/test_streamlit_integration.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_DIR = PROJECT_ROOT / "streamlit_app"
sys.path.insert(0, str(STREAMLIT_DIR))

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_STREAMLIT_INTEGRATION") != "1",
    reason="requires the local PostgreSQL application database",
)


@pytest.mark.parametrize("page_index", range(7))
def test_streamlit_page_renders_without_exception(page_index: int) -> None:
    app = AppTest.from_file(
        str(STREAMLIT_DIR / "app_main.py"),
        default_timeout=90,
    ).run()

    assert not app.exception
    assert app.radio

    app.radio[0].set_value(page_index)
    app.run()

    assert not app.exception
    assert not app.error
