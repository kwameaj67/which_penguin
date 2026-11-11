"""Predict a penguin's species with a trained model.

Usage:
    uv run predict --island Biscoe --sex Male \\
        --bill-length 48.6 --bill-depth 16.0 --flipper-length 230 --body-mass 5800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from which_penguin.train import DEFAULT_MODEL_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a penguin's species.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--island", choices=["Biscoe", "Dream", "Torgersen"])
    parser.add_argument("--sex", choices=["Male", "Female"])
    parser.add_argument("--bill-length", type=float, help="mm")
    parser.add_argument("--bill-depth", type=float, help="mm")
    parser.add_argument("--flipper-length", type=float, help="mm")
    parser.add_argument("--body-mass", type=float, help="grams")
    args = parser.parse_args()

    if not args.model.exists():
        parser.error(f"{args.model} not found. Run `uv run train` first.")

    # Any omitted measurement is passed as missing and imputed by the pipeline.
    row = pd.DataFrame(
        [
            {
                "bill_length_mm": args.bill_length,
                "bill_depth_mm": args.bill_depth,
                "flipper_length_mm": args.flipper_length,
                "body_mass_g": args.body_mass,
                "island": args.island,
                "sex": args.sex,
            }
        ]
    ).astype({"bill_length_mm": float, "bill_depth_mm": float,
              "flipper_length_mm": float, "body_mass_g": float})

    model = joblib.load(args.model)
    species = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]

    print(f"Predicted species: {species}")
    for name, p in sorted(zip(model.classes_, probabilities), key=lambda t: -t[1]):
        print(f"  {name:<10} {p:6.1%}")


if __name__ == "__main__":
    main()
