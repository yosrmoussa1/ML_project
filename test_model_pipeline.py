"""Tests unitaires du pipeline (exécutés par la task Prefect `tests`)."""

import numpy as np
import pandas as pd
import pytest

from model_pipeline import (
    COLUMNS,
    clean_data,
    evaluate_model,
    load_model,
    predict_model,
    save_model,
    train_model,
)


def _fake_raw_data(n=200, seed=0):
    """Petit jeu de données factice au format Adult (valeurs '?' -> NaN)."""
    r = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": r.integers(17, 90, n),
            "workclass": r.choice(["Private", "State-gov", np.nan], n),
            "fnlwgt": r.integers(10000, 500000, n),
            "education": "Bachelors",
            "education_num": r.integers(1, 16, n),
            "marital_status": r.choice(["Married-civ-spouse", "Never-married"], n),
            "occupation": r.choice(["Sales", "Craft-repair", np.nan], n),
            "relationship": r.choice(["Husband", "Own-child"], n),
            "race": r.choice(["White", "Black"], n),
            "sex": r.choice(["Male", "Female"], n),
            "capital_gain": r.choice([0, 0, 5000, 99999], n),
            "capital_loss": r.choice([0, 1500], n),
            "hours_per_week": r.integers(10, 80, n),
            "native_country": r.choice(["United-States", "Mexico", np.nan], n),
        }
    )
    score = df["education_num"] / 16 + (df["marital_status"] == "Married-civ-spouse")
    df["income"] = np.where(score + r.random(n) * 0.5 > 1.2, ">50K", "<=50K")
    return df[COLUMNS]


@pytest.fixture(scope="module")
def prepared():
    """Données nettoyées + split simple pour les tests."""
    df = clean_data(_fake_raw_data())
    X = df.drop(columns=["income_binary"])
    y = df["income_binary"]
    return X.iloc[:150], X.iloc[150:], y.iloc[:150], y.iloc[150:]


def test_clean_data_has_no_missing_values_and_binary_target():
    df = clean_data(_fake_raw_data())
    assert df.isna().sum().sum() == 0
    assert set(df["income_binary"].unique()) <= {0, 1}
    assert "fnlwgt" not in df.columns and "education" not in df.columns


def test_train_and_evaluate_return_metrics(prepared):
    X_train, X_test, y_train, y_test = prepared
    model = train_model(X_train, y_train, model_type="gb")
    metrics = evaluate_model(model, X_test, y_test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_train_model_rejects_unknown_type(prepared):
    X_train, _, y_train, _ = prepared
    with pytest.raises(ValueError):
        train_model(X_train, y_train, model_type="inconnu")


def test_save_and_load_roundtrip(prepared, tmp_path):
    X_train, X_test, y_train, _ = prepared
    model = train_model(X_train, y_train, model_type="knn")
    path = str(tmp_path / "m.joblib")
    save_model(model, path)
    loaded = load_model(path)
    assert (loaded.predict(X_test) == model.predict(X_test)).all()


def test_predict_model_output_format(prepared):
    X_train, X_test, y_train, _ = prepared
    model = train_model(X_train, y_train, model_type="gb")
    result = predict_model(model, X_test.head(3))
    assert list(result.columns) == ["prediction", "proba_>50K"]
    assert len(result) == 3
    assert set(result["prediction"]) <= {"<=50K", ">50K"}


def test_load_model_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_model("fichier_inexistant.joblib")
