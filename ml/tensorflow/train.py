"""TensorFlow learning example for demand prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATASET = Path(__file__).resolve().parents[1] / "forecasting" / "sample_demand.csv"


def main() -> None:
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is optional for this learning project. Install `tensorflow` to run this example.")
        return

    frame = pd.read_csv(DATASET)
    features = frame[["month_index", "price", "promotion_flag"]].astype("float32").values
    labels = frame["demand"].astype("float32").values

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(3,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(features, labels, epochs=10, verbose=0)
    output_path = Path(__file__).resolve().parent / "demand_model.keras"
    model.save(output_path)
    print(f"Saved TensorFlow example model to {output_path}")


if __name__ == "__main__":
    main()
