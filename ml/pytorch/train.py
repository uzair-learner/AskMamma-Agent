"""PyTorch learning example for demand prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET = Path(__file__).resolve().parents[1] / "forecasting" / "sample_demand.csv"


def main() -> None:
    try:
        import torch
        from torch import nn
    except ImportError:
        print("PyTorch is optional for this learning project. Install `torch` to run this example.")
        return

    frame = pd.read_csv(DATASET)
    features = torch.tensor(frame[["month_index", "price", "promotion_flag"]].values, dtype=torch.float32)
    labels = torch.tensor(frame["demand"].values, dtype=torch.float32).unsqueeze(1)

    model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()

    output_path = Path(__file__).resolve().parent / "demand_model.pt"
    torch.save(model.state_dict(), output_path)
    print(f"Saved PyTorch example model to {output_path}")


if __name__ == "__main__":
    main()
