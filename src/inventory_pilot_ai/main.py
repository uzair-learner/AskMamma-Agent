"""Application entry point used by uvicorn."""

from inventory_pilot_ai.api.routes import app

__all__ = ["app"]
