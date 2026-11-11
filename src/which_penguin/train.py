"""Compare models, then save the one with the best cross-validated accuracy.

Usage:
    uv run train [--output models/penguin_classifier.joblib]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from which_penguin.pipeline import compare_models, load_data, split_data

DEFAULT_MODEL_PATH = Path("models/penguin_classifier.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)

    results, searches = compare_models(X_train, y_train, X_test, y_test)
    print(results.round(4).to_string())

    best_name = results.index[0]  # highest CV accuracy
    model = searches[best_name].best_estimator_

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"\nSaved {best_name} to {args.output}")


if __name__ == "__main__":
    main()
