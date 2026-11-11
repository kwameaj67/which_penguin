"""Data loading and model construction for the penguin species classifier."""

from __future__ import annotations

import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "species"
NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# l1_ratio=0 is an L2 (ridge) penalty, l1_ratio=1 is L1 (lasso).
# scikit-learn >= 1.8 deprecates the `penalty` argument in favour of `l1_ratio`.
PARAM_GRID = {
    "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "classifier__l1_ratio": [0.0, 1.0],
}


def load_data() -> pd.DataFrame:
    """Load the Palmer Penguins dataset (downloaded by seaborn on first use)."""
    return sns.load_dataset("penguins")


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split into raw features and target."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Preprocessing + logistic regression in a single pipeline that accepts raw data."""
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
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # saga supports both L1 and L2 penalties, so one solver covers the whole grid.
    classifier = LogisticRegression(solver="saga", max_iter=5000, random_state=random_state)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def tune(X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5) -> GridSearchCV:
    """Cross-validated grid search over regularisation strength and type."""
    search = GridSearchCV(
        build_pipeline(), PARAM_GRID, cv=cv, scoring="accuracy", n_jobs=-1
    )
    return search.fit(X_train, y_train)
