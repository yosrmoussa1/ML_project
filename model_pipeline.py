"""
model_pipeline.py
-----------------
Fonctions modulaires du pipeline ML - Jeu de données Adult (Census Income).
Objectif : prédire si un individu gagne >50K $/an (classification binaire).

Étapes : prepare_data -> train_model -> evaluate_model -> save_model / load_model
(Modèles du notebook : KNN et Gradient Boosting.)
"""

import os
import zipfile

import joblib

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
COLS_MANQUANTES = ["workclass", "occupation", "native_country"]
COLS_A_ENCODER = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "age_group",
]
COLS_A_NORMALISER = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]


# --------------------------------------------------------------------------- #
# 1) Chargement + prétraitement
# --------------------------------------------------------------------------- #
def _trouver_fichier(dossier, nom):
    """Cherche récursivement `nom` dans `dossier`."""
    for racine, _, fichiers in os.walk(dossier):
        if nom in fichiers:
            return os.path.join(racine, nom)
    raise FileNotFoundError(f"'{nom}' introuvable dans {dossier}")


def load_raw_data(data_path="adult.zip", extract_dir="adult_data"):
    """
    Charge adult.data et adult.test (depuis un .zip ou un dossier) et les fusionne.

    Returns:
        pd.DataFrame: données brutes fusionnées (valeurs '?' -> NaN).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Chemin introuvable : {data_path}")

    if zipfile.is_zipfile(data_path):
        if not os.path.exists(extract_dir):
            with zipfile.ZipFile(data_path, "r") as z:
                z.extractall(extract_dir)
        dossier = extract_dir
    else:
        dossier = data_path  # on suppose un dossier

    kw = dict(header=None, names=COLUMNS, sep=r",\s*", engine="python", na_values="?")
    df_train = pd.read_csv(_trouver_fichier(dossier, "adult.data"), **kw)
    df_test = pd.read_csv(_trouver_fichier(dossier, "adult.test"), skiprows=1, **kw)

    # Le fichier de test a un point final sur l'étiquette ('>50K.')
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)
    return pd.concat([df_train, df_test], ignore_index=True)


def clean_data(df):
    """
    Nettoyage + feature engineering (identique au notebook) :
    imputation par le mode, suppression de fnlwgt/education, indicateurs
    has_capital_gain/loss, plafonnement de capital_gain, age_group, is_us,
    encodage one-hot, cible binaire.

    Returns:
        pd.DataFrame: données encodées (non normalisées) avec `income_binary`.
    """
    df = df.copy()

    # Valeurs manquantes -> mode
    for col in COLS_MANQUANTES:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Colonnes redondantes / non explicatives
    df = df.drop(columns=["fnlwgt", "education"])

    # Capital : indicateurs binaires + plafonnement du plateau 99999
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    cap_99 = df.loc[df["capital_gain"] < 99999, "capital_gain"].quantile(0.99)
    df["capital_gain"] = df["capital_gain"].clip(upper=cap_99)

    # Feature engineering
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=["<=25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    df["is_us"] = (df["native_country"] == "United-States").astype(int)
    df = df.drop(columns=["native_country"])

    # Cible + encodage
    df["income_binary"] = (df["income"] == ">50K").astype(int)
    df = pd.get_dummies(df, columns=COLS_A_ENCODER, drop_first=True)
    df = df.drop(columns=["income"])

    # Toutes les colonnes en numérique (les dummies peuvent être booléennes)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def prepare_data(
    data_path="adult.zip",
    test_size=0.25,
    random_state=RANDOM_STATE,
    save_clean_csv=None,
):
    """
    Charge et prétraite les données, puis découpe en train/test (stratifié).
    Le StandardScaler est ajusté sur le train uniquement (pas de fuite).

    Args:
        data_path (str): adult.zip, dossier contenant adult.data/adult.test,
                         ou CSV déjà nettoyé contenant `income_binary`.
        test_size (float): proportion du jeu de test.
        random_state (int): graine aléatoire.
        save_clean_csv (str|None): si fourni, enregistre le jeu encodé.

    Returns:
        tuple: X_train, X_test, y_train, y_test, scaler
    """
    if data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        if "income_binary" not in df.columns:
            raise ValueError("Le CSV doit contenir la colonne 'income_binary'.")
    else:
        df = clean_data(load_raw_data(data_path))

    if save_clean_csv:
        df.to_csv(save_clean_csv, index=False)

    X = df.drop(columns=["income_binary"])
    y = df["income_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[COLS_A_NORMALISER] = scaler.fit_transform(X_train[COLS_A_NORMALISER])
    X_test[COLS_A_NORMALISER] = scaler.transform(X_test[COLS_A_NORMALISER])

    return X_train, X_test, y_train, y_test, scaler


# --------------------------------------------------------------------------- #
# 2) Entraînement
# --------------------------------------------------------------------------- #
def find_best_k(X_train, y_train, k_max=15, cv_splits=5, random_state=RANDOM_STATE):
    """
    Choisit k (KNN) par validation croisée sur le TRAIN (et non sur le test,
    ce qui était la limite signalée dans le notebook).
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scores = {
        k: cross_val_score(
            KNeighborsClassifier(n_neighbors=k),
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
        ).mean()
        for k in range(1, k_max)
    }
    best_k = max(scores, key=scores.get)
    print(f"Meilleur k (CV, ROC-AUC) = {best_k} ({scores[best_k]:.3f})")
    return best_k


def train_model(
    X_train, y_train, model_type="gb", n_neighbors=12, random_state=RANDOM_STATE
):
    """
    Entraîne un modèle.

    Args:
        model_type (str): 'gb' (Gradient Boosting) ou 'knn'.
        n_neighbors (int): k pour le KNN (12 = valeur retenue dans le notebook).

    Returns:
        modèle scikit-learn entraîné.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train n'ont pas la même taille.")

    if model_type == "gb":
        model = GradientBoostingClassifier(random_state=random_state)
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        raise ValueError("model_type doit être 'gb' ou 'knn'.")

    model.fit(X_train, y_train)
    return model


# --------------------------------------------------------------------------- #
# 3) Évaluation
# --------------------------------------------------------------------------- #
def evaluate_model(model, X_test, y_test, name="Modèle"):
    """
    Évalue le modèle (mêmes métriques que le notebook) et les affiche.

    Returns:
        dict: accuracy, precision, recall, f1, roc_auc, avg_precision.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    resultats = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
    }

    print(f"\n=== {name} ===")
    for k, v in resultats.items():
        print(f"{k:<14}: {v:.3f}")
    print("\nMatrice de confusion (lignes = réel, colonnes = prédit) :")
    print(confusion_matrix(y_test, y_pred))
    print(
        "\n",
        classification_report(
            y_test, y_pred, zero_division=0, target_names=["<=50K", ">50K"]
        ),
    )
    return resultats


def feature_importance(model, feature_names, top=15):
    """Importance des variables (modèles à arbres uniquement, ex. Gradient Boosting)."""
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Ce modèle n'expose pas feature_importances_ (ex. KNN).")
    return (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(top)
    )


# --------------------------------------------------------------------------- #
# 4) Sauvegarde / chargement
# --------------------------------------------------------------------------- #
def save_model(model, path="model.joblib", scaler=None):
    """Sauvegarde le modèle (et optionnellement le scaler) avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")
    if scaler is not None:
        scaler_path = os.path.splitext(path)[0] + "_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"Scaler sauvegardé : {scaler_path}")


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé avec joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return joblib.load(path)


def predict_model(model, X):
    """
    Prédit la classe de revenu pour de nouvelles observations.

    Args:
        model: modèle entraîné.
        X (pd.DataFrame): observations prétraitées (mêmes colonnes que le train).

    Returns:
        pd.DataFrame: colonnes `prediction` ('<=50K' / '>50K') et `proba_>50K`.
    """
    if len(X) == 0:
        raise ValueError("X est vide : rien à prédire.")
    classes = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "prediction": ["<=50K" if c == 0 else ">50K" for c in classes],
            "proba_>50K": proba.round(3),
        },
        index=X.index,
    )
