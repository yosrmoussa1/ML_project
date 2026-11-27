"""Tests de l'API FastAPI (exécutés par la task Prefect `tests`)."""

import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from app import app
from model_pipeline import (
    COLS_A_NORMALISER,
    INPUT_COLUMNS,
    clean_data,
    encode_raw_input,
    save_model,
    train_model,
)
from test_model_pipeline import _fake_raw_data

EXEMPLE = {
    "age": 39,
    "workclass": "State-gov",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Sales",
    "relationship": "Own-child",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Entraîne un petit modèle sur données factices et le sauvegarde."""
    raw = _fake_raw_data(400)
    df = clean_data(raw)
    X, y = df.drop(columns=["income_binary"]), df["income_binary"]
    scaler = StandardScaler()
    scaler.capital_gain_cap_ = float(df["capital_gain"].max())
    X = X.copy()
    X[COLS_A_NORMALISER] = scaler.fit_transform(X[COLS_A_NORMALISER])
    model = train_model(X, y, model_type="gb")
    path = str(tmp_path_factory.mktemp("api") / "model.joblib")
    save_model(model, path, scaler=scaler)
    return path, model, scaler, raw, X


@pytest.fixture()
def client(trained, monkeypatch):
    """Client de test avec le modèle chargé au démarrage (lifespan)."""
    monkeypatch.setenv("MODEL_PATH", trained[0])
    with TestClient(app) as c:
        yield c


def test_health_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction(client):
    r = client.post("/predict", json=EXEMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("<=50K", ">50K")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] == int(body["label"] == ">50K")


def test_predict_rejects_invalid_category(client):
    r = client.post("/predict", json={**EXEMPLE, "sex": "Inconnu"})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplet = {k: v for k, v in EXEMPLE.items() if k != "age"}
    assert client.post("/predict", json=incomplet).status_code == 422


def test_predict_without_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "absent.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=EXEMPLE).status_code == 503


def test_encoding_matches_training_pipeline(trained):
    """L'encodage de l'API doit reproduire clean_data() + normalisation."""
    _, model, scaler, raw, X_train = trained
    complet = raw.dropna(subset=["workclass", "occupation", "native_country"])
    lignes = complet.head(20)
    attendu = X_train.loc[lignes.index]
    obtenu = encode_raw_input(
        lignes[INPUT_COLUMNS].to_dict("records"),
        list(model.feature_names_in_),
        scaler,
    )
    obtenu.index = lignes.index
    assert list(obtenu.columns) == list(attendu.columns)
    assert ((obtenu - attendu).abs().max().max()) < 1e-9


@pytest.fixture()
def retrain_env(tmp_path, monkeypatch):
    """Données factices (CSV nettoyé) + chemin de modèle temporaire."""
    csv = tmp_path / "clean.csv"
    clean_data(_fake_raw_data(400)).to_csv(csv, index=False)
    monkeypatch.setenv("DATA_PATH", str(csv))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    return tmp_path


def test_retrain_new_hyperparameters(retrain_env):
    params = {"model_type": "gb", "n_estimators": 30, "max_depth": 2}
    with TestClient(app) as c:
        r = c.post("/retrain", json=params)
        assert r.status_code == 200
        body = r.json()
        assert body["hyperparameters"] == {"n_estimators": 30, "max_depth": 2}
        assert 0.0 <= body["metrics"]["accuracy"] <= 1.0
        assert (retrain_env / "model.joblib").exists()
        assert (retrain_env / "model_scaler.joblib").exists()
        # le nouveau modèle est utilisable immédiatement
        assert c.post("/predict", json=EXEMPLE).status_code == 200


def test_retrain_knn(retrain_env):
    with TestClient(app) as c:
        r = c.post("/retrain", json={"model_type": "knn", "n_neighbors": 5})
        assert r.status_code == 200
        assert r.json()["hyperparameters"] == {"n_neighbors": 5}


def test_retrain_rejects_invalid_hyperparameter(retrain_env):
    with TestClient(app) as c:
        assert c.post("/retrain", json={"n_estimators": 1}).status_code == 422


def test_retrain_missing_data_returns_404(retrain_env, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(retrain_env / "absent.csv"))
    with TestClient(app) as c:
        assert c.post("/retrain", json={}).status_code == 404
