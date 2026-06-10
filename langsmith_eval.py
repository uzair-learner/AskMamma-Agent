"""Compatibility entrypoint for running the local agent evaluation.

LangSmith tracing can be enabled through `.env`, but the canonical evaluation
script for this Inventory Pilot AI system is `scripts/evaluate_agent.py`.
"""

from __future__ import annotations

from scripts.evaluate_agent import main


if __name__ == "__main__":
    main()
