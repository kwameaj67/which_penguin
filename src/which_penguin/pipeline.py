"""Data loading, model definitions and model comparison for the penguin classifier."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Each model and the hyperparameters to try for it.
MODELS = {
    "Logistic regression": (
        # saga supports both L1 and L2 penalties.
        LogisticRegression(solver="saga", max_iter=5000, random_state=RANDOM_STATE),
        # l1_ratio=0 is an L2 penalty, l1_ratio=1 is L1.
        {"C": [0.01, 0.1, 1, 10, 100], "l1_ratio": [0.0, 1.0]},
    ),
    "K-nearest neighbours": (
        KNeighborsClassifier(),
        {"n_neighbors": [1, 3, 5, 7, 9, 15], "weights": ["uniform", "distance"]},
    ),
    "Decision tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [2, 3, 4, 5, None], "min_samples_leaf": [1, 5, 10]},
    ),
    "Random forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 300], "max_depth": [3, 5, None]},
    ),
}


def load_data() -> pd.DataFrame:
    """Load the Palmer Penguins dataset (downloaded by seaborn on first use)."""
    return sns.load_dataset("penguins")


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified train/test split into raw features and target."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def build_pipeline(classifier) -> Pipeline:
    """Preprocessing + classifier in a single pipeline that accepts raw data."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def tune(classifier, param_grid: dict, X_train, y_train) -> GridSearchCV:
    """Grid search one model with 5-fold cross-validation."""
    # Grid keys need a "classifier__" prefix to reach the step inside the pipeline.
    grid = {f"classifier__{name}": values for name, values in param_grid.items()}

    search = GridSearchCV(
        build_pipeline(classifier),
        grid,
        cv=5,
        scoring={"accuracy": "accuracy", "recall": "recall_macro"},
        refit="accuracy",  # pick the best settings by accuracy
    )
    return search.fit(X_train, y_train)


def compare_models(X_train, y_train, X_test, y_test):
    """Tune every model, then score it with cross-validation and on the test set."""
    rows = []
    searches = {}

    for name, (classifier, param_grid) in MODELS.items():
        search = tune(classifier, param_grid, X_train, y_train)
        best = search.best_index_
        y_pred = search.predict(X_test)
        # Drop the "classifier__" prefix so the table is easier to read.
        best_params = {k.removeprefix("classifier__"): v for k, v in search.best_params_.items()}
       
        rows.append({
            "model": name,
            "cv_accuracy": search.cv_results_["mean_test_accuracy"][best],
            "cv_recall": search.cv_results_["mean_test_recall"][best],
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred, average="macro"),
            "best_params": best_params,
        })
        searches[name] = search

    results = pd.DataFrame(rows).set_index("model")
    # A stable sort keeps the MODELS order on ties, so the simpler model wins.
    results = results.sort_values("cv_accuracy", ascending=False, kind="stable")
    return results, searches
