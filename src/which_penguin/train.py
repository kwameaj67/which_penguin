"""Train, evaluate and save the penguin classifier.

Usage:
    uv run train [--output models/penguin_classifier.joblib]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report

from which_penguin.pipeline import load_data, split_data, tune

DEFAULT_MODEL_PATH = Path("models/penguin_classifier.joblib")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    data = load_data()
    
    X_train, X_test, y_train, y_test = split_data(data)
    search = tune(X_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    
    print(f"Best params:      {search.best_params_}")
    print(f"CV accuracy:      {search.best_score_:.4f}")
    print(f"Test accuracy:    {accuracy_score(y_test, y_pred):.4f}\n")
    print(classification_report(y_test, y_pred))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
