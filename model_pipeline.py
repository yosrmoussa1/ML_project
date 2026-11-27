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
AGE_BINS = [0, 25, 35, 45, 55, 65, 120]
AGE_LABELS = ["<=25", "26-35", "36-45", "46-55", "56-65", "65+"]
# Variables brutes attendues par l'API (avant encodage), cf. adult.data
INPUT_COLUMNS = [
    "age",
    "workclass",
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
        bins=AGE_BINS,
        labels=AGE_LABELS,
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
    # Plafond appliqué à capital_gain pendant le nettoyage : il est sauvegardé
    # avec le scaler pour que l'API reproduise exactement le même prétraitement.
    scaler.capital_gain_cap_ = float(df["capital_gain"].max())
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[COLS_A_NORMALISER] = scaler.fit_transform(X_train[COLS_A_NORMALISER])
    X_test[COLS_A_NORMALISER] = scaler.transform(X_test[COLS_A_NORMALISER])

    return X_train, X_test, y_train, y_test, scaler


# --------------------------------------------------------------------------- #
# 2) Entraînement
# --------------------------------------------------------------------------- #
def retrain(
    data_path="adult.zip",
    model_path="model.joblib",
    model_type="gb",
    hyperparams=None,
):
    """
    Réentraîne le modèle avec de nouveaux hyperparamètres, l'évalue puis
    remplace le modèle et le scaler sauvegardés.

    Args:
        data_path (str): données d'entraînement (cf. prepare_data).
        model_path (str): chemin du modèle à écraser.
        model_type (str): 'gb' ou 'knn'.
        hyperparams (dict|None): hyperparamètres du modèle.

    Returns:
        tuple: (model, scaler, metrics) avec metrics = dict de evaluate_model().
    """
    X_train, X_test, y_train, y_test, scaler = prepare_data(data_path)
    model = train_model(X_train, y_train, model_type, **(hyperparams or {}))
    metrics = evaluate_model(model, X_test, y_test, name="Modèle réentraîné")
    save_model(model, model_path, scaler=scaler)
    return model, scaler, {k: float(v) for k, v in metrics.items()}


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
    X_train,
    y_train,
    model_type="gb",
    n_neighbors=12,
    random_state=RANDOM_STATE,
    **hyperparams,
):
    """
    Entraîne un modèle.

    Args:
        model_type (str): 'gb' (Gradient Boosting) ou 'knn'.
        n_neighbors (int): k pour le KNN (12 = valeur retenue dans le notebook).
        **hyperparams: hyperparamètres supplémentaires transmis au modèle
            (ex. n_estimators, learning_rate, max_depth pour 'gb').

    Returns:
        modèle scikit-learn entraîné.
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train n'ont pas la même taille.")

    if model_type == "gb":
        model = GradientBoostingClassifier(random_state=random_state, **hyperparams)
    elif model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=n_neighbors, **hyperparams)
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
# 4) Prédiction
# --------------------------------------------------------------------------- #
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


def encode_raw_input(records, feature_columns, scaler):
    """
    Transforme des observations BRUTES (ex. JSON reçu par l'API) en variables
    prêtes pour le modèle, en reproduisant clean_data() + la normalisation.

    Args:
        records (dict | list[dict]): une ou plusieurs observations contenant
            les clés de INPUT_COLUMNS.
        feature_columns (list[str]): colonnes vues à l'entraînement
            (ex. model.feature_names_in_).
        scaler (StandardScaler): scaler ajusté lors de prepare_data().

    Returns:
        pd.DataFrame: observations encodées, colonnes dans l'ordre attendu.
    """
    if isinstance(records, dict):
        records = [records]
    raw = pd.DataFrame(records)
    manquantes = [c for c in INPUT_COLUMNS if c not in raw.columns]
    if manquantes:
        raise ValueError(f"Variables manquantes : {manquantes}")

    X = pd.DataFrame(0.0, index=raw.index, columns=list(feature_columns))

    # Variables numériques + indicateurs (calculés AVANT le plafonnement)
    capital_gain = raw["capital_gain"].astype(float)
    cap = getattr(scaler, "capital_gain_cap_", None)
    X["has_capital_gain"] = (capital_gain > 0).astype(float)
    X["has_capital_loss"] = (raw["capital_loss"] > 0).astype(float)
    X["capital_gain"] = capital_gain if cap is None else capital_gain.clip(upper=cap)
    for col in ("age", "education_num", "capital_loss", "hours_per_week"):
        X[col] = raw[col].astype(float)
    X["is_us"] = (raw["native_country"] == "United-States").astype(float)

    # Variables catégorielles -> colonnes one-hot (la modalité de référence,
    # supprimée par drop_first, correspond à toutes les colonnes à 0)
    cat = raw.copy()
    cat["age_group"] = pd.cut(cat["age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    for col in COLS_A_ENCODER:
        for idx, val in cat[col].items():
            nom = f"{col}_{val}"
            if nom in X.columns:
                X.loc[idx, nom] = 1.0

    X[COLS_A_NORMALISER] = scaler.transform(X[COLS_A_NORMALISER])
    return X


# --------------------------------------------------------------------------- #
# 5) Sauvegarde / chargement
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
