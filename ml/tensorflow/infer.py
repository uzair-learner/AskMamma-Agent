"""TensorFlow inference example."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent / "demand_model.keras"
DATASET = Path(__file__).resolve().parents[1] / "forecasting" / "sample_demand.csv"


def main() -> None:
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is optional for this learning project. Install `tensorflow` to run this example.")
        return

    if not MODEL_PATH.exists():
        print("Run ml/tensorflow/train.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    row = pd.read_csv(DATASET).iloc[-1]
    sample = [[float(row["month_index"]), float(row["price"]), float(row["promotion_flag"])]]
    prediction = model.predict(sample, verbose=0)[0][0]
    print(f"Predicted demand: {prediction:.2f}")


if __name__ == "__main__":
    main()
