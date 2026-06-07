"""PyTorch inference example."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "demand_model.pt"
DATASET = Path(__file__).resolve().parents[1] / "forecasting" / "sample_demand.csv"


def main() -> None:
    try:
        import torch
        from torch import nn
    except ImportError:
        print("PyTorch is optional for this learning project. Install `torch` to run this example.")
        return

    if not MODEL_PATH.exists():
        print("Run ml/pytorch/train.py first.")
        return

    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    row = pd.read_csv(DATASET).iloc[-1]
    sample = torch.tensor([[float(row["month_index"]), float(row["price"]), float(row["promotion_flag"])]], dtype=torch.float32)
    prediction = model(sample).item()
    print(f"Predicted demand: {prediction:.2f}")


if __name__ == "__main__":
    main()
