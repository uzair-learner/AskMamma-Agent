"""Compatibility wrapper for uvicorn backend:app."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from inventory_pilot_ai.main import app
