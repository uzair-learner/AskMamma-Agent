"""Compatibility wrapper for running the Streamlit UI from the project root."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from inventory_pilot_ai.ui.app import *  # noqa: F401,F403
