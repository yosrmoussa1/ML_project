"""
pipeline_prefect.py
-------------------
Orchestration du pipeline ML avec Prefect (tasks + flows).

Flows : all, train, evaluate, predict, code, install, fetch
Lancement :
    python pipeline_prefect.py --flow all
    python pipeline_prefect.py --flow train        (alias : entrainement)
    python pipeline_prefect.py --flow evaluate
    python pipeline_prefect.py --flow code
"""

import argparse

from prefect import flow, get_run_logger, task

import main as steps


# --------------------------------------------------------------------------- #
# TASKS (chacune appelle une fonction de main.py)
# --------------------------------------------------------------------------- #
@task(name="Récupérer le projet (git)")
def fetch_task(repo_url=None):
    steps.fetch_project(repo_url)


@task(name="Installer les dépendances", retries=2, retry_delay_seconds=10)
def install_task():
    steps.install_dependencies()


@task(name="Formatage du code (black)")
def format_task():
    steps.format_code()


@task(name="Qualité du code (flake8)")
def quality_task():
    steps.check_quality()


@task(name="Sécurité du code (bandit)")
def security_task():
    steps.check_security()


@task(name="Tests unitaires (pytest)")
def tests_task():
    steps.run_unit_tests()


@task(name="Préparation des données")
def prepare_task(data_path):
    return steps.run_prepare(data_path)


@task(name="Entraînement du modèle")
def train_task(X_train, y_train, model_type, k, tune_k):
    return steps.run_train(X_train, y_train, model_type, k, tune_k)


@task(name="Sauvegarder le modèle")
def save_task(model, model_path, scaler):
    steps.run_save(model, model_path, scaler)


@task(name="Charger le modèle")
def load_task(model_path):
    return steps.run_load(model_path)


@task(name="Évaluer le modèle")
def evaluate_task(model, X_test, y_test):
    return steps.run_evaluate(model, X_test, y_test)


@task(name="Prédiction")
def predict_task(model, X_test, y_test):
    return steps.run_predict(model, X_test, y_test)


# --------------------------------------------------------------------------- #
# FLOWS
# --------------------------------------------------------------------------- #
@flow(name="fetch")
def fetch_flow(repo_url: str | None = None):
    """Récupération du projet depuis le dépôt Git."""
    fetch_task(repo_url)


@flow(name="install")
def install_flow():
    """Installation des dépendances."""
    install_task()


@flow(name="code")
def code_flow(skip_install: bool = False):
    """Suivi du code : dépendances, formatage, qualité, sécurité, tests."""
    if not skip_install:
        install_task()
    format_task()
    quality_task()
    security_task()
    tests_task()


@flow(name="train")
def train_flow(
    data_path: str = "adult.zip",
    model_path: str = "model.joblib",
    model_type: str = "gb",
    k: int = 12,
    tune_k: bool = False,
):
    """Préparation des données + entraînement (+ sauvegarde du modèle)."""
    X_train, X_test, y_train, y_test, scaler = prepare_task(data_path)
    model = train_task(X_train, y_train, model_type, k, tune_k)
    save_task(model, model_path, scaler)
    return model


@flow(name="evaluate")
def evaluate_flow(data_path: str = "adult.zip", model_path: str = "model.joblib"):
    """Chargement du modèle + évaluation."""
    _, X_test, _, y_test, _ = prepare_task(data_path)
    model = load_task(model_path)
    return evaluate_task(model, X_test, y_test)


@flow(name="predict")
def predict_flow(data_path: str = "adult.zip", model_path: str = "model.joblib"):
    """Chargement du modèle + prédiction sur quelques observations du test."""
    _, X_test, _, y_test, _ = prepare_task(data_path)
    model = load_task(model_path)
    return predict_task(model, X_test, y_test)


@flow(name="all")
def all_flow(
    data_path: str = "adult.zip",
    model_path: str = "model.joblib",
    model_type: str = "gb",
    k: int = 12,
    tune_k: bool = False,
    skip_install: bool = False,
    repo_url: str | None = None,
):
    """
    Pipeline complet : (git) -> dépendances -> formatage -> qualité -> sécurité
    -> tests -> préparation -> entraînement -> sauvegarde -> évaluation
    -> prédiction.
    """
    logger = get_run_logger()
    if repo_url:
        fetch_task(repo_url)
    code_flow(skip_install=skip_install)

    X_train, X_test, y_train, y_test, scaler = prepare_task(data_path)
    model = train_task(X_train, y_train, model_type, k, tune_k)
    save_task(model, model_path, scaler)
    metrics = evaluate_task(model, X_test, y_test)
    predict_task(model, X_test, y_test)
    logger.info("ROC-AUC final : %.3f", metrics["roc_auc"])
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Pipeline Prefect - Adult Income")
    p.add_argument(
        "--flow",
        default="all",
        choices=[
            "all",
            "train",
            "entrainement",
            "evaluate",
            "predict",
            "code",
            "install",
            "fetch",
        ],
        help="Flow à exécuter",
    )
    p.add_argument("--data", default="adult.zip")
    p.add_argument("--model", default="model.joblib")
    p.add_argument("--model-type", choices=["gb", "knn"], default="gb")
    p.add_argument("--skip-install", action="store_true")
    p.add_argument("--repo-url", default=None, help="URL du dépôt Git distant")
    return p.parse_args()


def main():
    a = parse_args()
    if a.flow == "all":
        all_flow(
            a.data,
            a.model,
            a.model_type,
            skip_install=a.skip_install,
            repo_url=a.repo_url,
        )
    elif a.flow in ("train", "entrainement"):
        train_flow(a.data, a.model, a.model_type)
    elif a.flow == "evaluate":
        evaluate_flow(a.data, a.model)
    elif a.flow == "predict":
        predict_flow(a.data, a.model)
    elif a.flow == "code":
        code_flow(skip_install=a.skip_install)
    elif a.flow == "install":
        install_flow()
    elif a.flow == "fetch":
        fetch_flow(a.repo_url)


if __name__ == "__main__":
    main()
