import numpy as np
import pandas as pd
import pytest

from which_penguin.pipeline import FEATURES, build_pipeline, load_data, split_data


@pytest.fixture(scope="module")
def data():
    return split_data(load_data())


@pytest.fixture(scope="module")
def fitted(data):
    X_train, _, y_train, _ = data
    return build_pipeline().fit(X_train, y_train)


def test_split_is_stratified(data):
    _, _, y_train, y_test = data
    train_share = y_train.value_counts(normalize=True)
    test_share = y_test.value_counts(normalize=True)
    assert (train_share - test_share).abs().max() < 0.05


def test_test_accuracy_above_baseline(fitted, data):
    _, X_test, _, y_test = data
    assert fitted.score(X_test, y_test) > 0.95


def test_handles_missing_values(fitted):
    row = pd.DataFrame([{f: np.nan for f in FEATURES}]).astype(
        {"island": object, "sex": object}
    )
    assert fitted.predict(row)[0] in fitted.classes_


def test_handles_unseen_category(fitted, data):
    _, X_test, _, _ = data
    row = X_test.iloc[[0]].copy()
    row["island"] = "Antarctica"
    assert fitted.predict(row)[0] in fitted.classes_
