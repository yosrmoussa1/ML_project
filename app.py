"""
app.py
------
Service REST FastAPI exposant la fonction de prédiction du modèle
(Adult Census Income : prédire si un individu gagne >50K $/an).

Lancement :
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    python pipeline_prefect.py --flow api

Documentation interactive (Swagger) : http://127.0.0.1:8000/docs

Exemple de requête :
    curl -X POST http://127.0.0.1:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{"age": 39, "workclass": "State-gov", "education_num": 13,
           "marital_status": "Never-married", "occupation": "Adm-clerical",
           "relationship": "Not-in-family", "race": "White", "sex": "Male",
           "capital_gain": 2174, "capital_loss": 0, "hours_per_week": 40,
           "native_country": "United-States"}'

Réentraînement avec de nouveaux hyperparamètres :
    curl -X POST http://127.0.0.1:8000/retrain \\
      -H "Content-Type: application/json" \\
      -d '{"model_type": "gb", "n_estimators": 200, "learning_rate": 0.1,
           "max_depth": 4}'

Variables d'environnement :
    MODEL_PATH : chemin du modèle joblib (défaut : model.joblib). Le scaler
                 est cherché à côté : <nom>_scaler.joblib.
    DATA_PATH  : données utilisées par /retrain (défaut : adult.zip).
"""

import os
import threading
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_pipeline import encode_raw_input, load_model, predict_model, retrain

DEFAULT_MODEL_PATH = "model.joblib"
DEFAULT_DATA_PATH = "adult.zip"
retrain_lock = threading.Lock()  # un seul réentraînement à la fois

# Objets chargés au démarrage (modèle + scaler), partagés par les routes
artifacts = {"model": None, "scaler": None}


def scaler_path_for(model_path):
    """Chemin du scaler associé au modèle (même convention que save_model)."""
    return os.path.splitext(model_path)[0] + "_scaler.joblib"


@asynccontextmanager
async def lifespan(_app):
    """Charge le modèle entraîné depuis le disque au démarrage de l'API."""
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    artifacts["model"] = artifacts["scaler"] = None
    try:
        artifacts["model"] = load_model(model_path)
        artifacts["scaler"] = load_model(scaler_path_for(model_path))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError as exc:
        print(f"[ATTENTION] {exc} -> lancez d'abord : python main.py --train --save")
    yield
    artifacts["model"] = artifacts["scaler"] = None


app = FastAPI(
    title="API de prédiction - Adult Income",
    description="Prédit si le revenu annuel d'un individu dépasse 50K $.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schémas d'entrée / sortie
# --------------------------------------------------------------------------- #
class IncomeFeatures(BaseModel):
    """Caractéristiques brutes d'un individu (colonnes du jeu Adult)."""

    age: int = Field(..., ge=0, le=120, description="Âge en années")
    workclass: Literal[
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ]
    education_num: int = Field(..., ge=1, le=16, description="Niveau d'études (1-16)")
    marital_status: Literal[
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ]
    occupation: Literal[
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ]
    relationship: Literal[
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ]
    race: Literal["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: float = Field(0, ge=0, description="Gains en capital")
    capital_loss: float = Field(0, ge=0, description="Pertes en capital")
    hours_per_week: int = Field(..., ge=1, le=99, description="Heures par semaine")
    native_country: str = Field("United-States", description="Pays d'origine")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "education_num": 13,
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Résultat de la prédiction."""

    prediction: int = Field(..., description="1 si revenu >50K, 0 sinon")
    label: Literal["<=50K", ">50K"]
    probability: float = Field(..., description="Probabilité de la classe >50K")


class RetrainRequest(BaseModel):
    """Hyperparamètres du réentraînement (champs omis = valeurs par défaut)."""

    model_type: Literal["gb", "knn"] = Field("gb", description="Type de modèle")
    n_estimators: Optional[int] = Field(None, ge=10, le=1000, description="gb")
    learning_rate: Optional[float] = Field(None, gt=0, le=1, description="gb")
    max_depth: Optional[int] = Field(None, ge=1, le=10, description="gb")
    n_neighbors: Optional[int] = Field(None, ge=1, le=50, description="knn")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_type": "gb",
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "max_depth": 4,
                }
            ]
        }
    }

    def hyperparams(self):
        """Hyperparamètres renseignés et pertinents pour le type de modèle."""
        noms = (
            ["n_estimators", "learning_rate", "max_depth"]
            if self.model_type == "gb"
            else ["n_neighbors"]
        )
        valeurs = {n: getattr(self, n) for n in noms}
        return {n: v for n, v in valeurs.items() if v is not None}


class RetrainResponse(BaseModel):
    """Résultat du réentraînement."""

    status: str
    model_type: str
    hyperparameters: dict
    metrics: dict


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", tags=["info"])
def root():
    """Message d'accueil."""
    return {"message": "API Adult Income : voir /docs", "predict": "POST /predict"}


@app.get("/health", tags=["info"])
def health():
    """Indique si le modèle est chargé et prêt."""
    return {"status": "ok", "model_loaded": artifacts["model"] is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(features: IncomeFeatures):
    """
    Effectue une prédiction à partir des caractéristiques fournies.

    Args:
        features (IncomeFeatures): caractéristiques brutes de l'individu.

    Returns:
        dict: prediction (0/1), label ('<=50K'/'>50K') et probabilité.

    Raises:
        HTTPException: 503 si le modèle n'est pas chargé, 500 si la prédiction échoue.
    """
    model, scaler = artifacts["model"], artifacts["scaler"]
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé : vérifiez MODEL_PATH et le fichier .joblib.",
        )
    try:
        X = encode_raw_input(
            features.model_dump(), list(model.feature_names_in_), scaler
        )
        result = predict_model(model, X).iloc[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erreur pendant la prédiction : {exc}"
        ) from exc

    label = result["prediction"]
    return PredictionResponse(
        prediction=int(label == ">50K"),
        label=label,
        probability=float(result["proba_>50K"]),
    )


@app.post("/retrain", response_model=RetrainResponse, tags=["training"])
def retrain_endpoint(params: RetrainRequest):
    """
    Réentraîne le modèle avec de nouveaux hyperparamètres (requête POST).

    Le modèle et le scaler sont évalués puis remplacés sur le disque, et
    l'API utilise immédiatement le nouveau modèle.

    Raises:
        HTTPException: 404 si les données sont introuvables, 409 si un
            réentraînement est déjà en cours, 500 en cas d'échec.
    """
    if not retrain_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Réentraînement déjà en cours.")
    try:
        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        data_path = os.getenv("DATA_PATH", DEFAULT_DATA_PATH)
        hyperparams = params.hyperparams()
        model, scaler, metrics = retrain(
            data_path, model_path, params.model_type, hyperparams
        )
        artifacts["model"], artifacts["scaler"] = model, scaler
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail=f"Données introuvables : {exc}"
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Erreur pendant le réentraînement : {exc}"
        ) from exc
    finally:
        retrain_lock.release()
    return RetrainResponse(
        status="Modèle réentraîné et rechargé",
        model_type=params.model_type,
        hyperparameters=hyperparams,
        metrics=metrics,
    )
