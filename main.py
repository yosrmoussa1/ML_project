"""
main.py
-------
Fonctions d'exécution des étapes du projet + interface CLI.
Ces fonctions sont réutilisées par les tasks Prefect (pipeline_prefect.py).

Exemples CLI :
    python main.py --prepare
    python main.py --train --save
    python main.py --train --model-type knn --tune-k --save --model knn.joblib
    python main.py --evaluate
    python main.py --predict
    python main.py --all
"""

import argparse
import os

# Commandes fixes, sans entrée utilisateur : usage de subprocess maîtrisé
import subprocess  # nosec B404
import sys

from model_pipeline import (
    evaluate_model,
    feature_importance,
    find_best_k,
    load_model,
    predict_model,
    prepare_data,
    save_model,
    train_model,
)

# Seuls ces fichiers sont analysés (qualité / sécurité) : plus rapide
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
TEST_FILE = "test_model_pipeline.py"


# --------------------------------------------------------------------------- #
# Étapes liées à l'environnement et au suivi du code
# --------------------------------------------------------------------------- #
def run_command(cmd):
    """Exécute une commande (liste d'arguments) et lève une erreur si échec."""
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, check=False)  # nosec B603
    if result.returncode != 0:
        raise RuntimeError(f"Commande en échec (code {result.returncode}) : {cmd}")


def fetch_project(repo_url=None, dest="ml_project_repo"):
    """
    Récupère le projet depuis le dépôt Git distant.
    - Si le dossier courant est déjà un dépôt Git : `git pull`.
    - Sinon, si `repo_url` est fourni : `git clone repo_url dest`.
    """
    if os.path.isdir(".git"):
        run_command(["git", "pull"])
    elif repo_url:
        run_command(["git", "clone", repo_url, dest])
    else:
        print("Pas de dépôt Git ni d'URL fournie : récupération ignorée.")


def install_dependencies(requirements="requirements.txt"):
    """Installe les dépendances listées dans requirements.txt."""
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--default-timeout=300",
            "--retries",
            "10",
            "-r",
            requirements,
        ]
    )


def format_code():
    """Formate le code avec black (uniquement les fichiers du projet)."""
    run_command([sys.executable, "-m", "black", *PROJECT_FILES, TEST_FILE])


def check_quality():
    """Vérifie la qualité du code avec flake8."""
    run_command(
        [
            sys.executable,
            "-m",
            "flake8",
            "--max-line-length=88",
            *PROJECT_FILES,
            TEST_FILE,
        ]
    )


def check_security():
    """Analyse la sécurité du code avec bandit."""
    run_command([sys.executable, "-m", "bandit", "-q", *PROJECT_FILES])


def run_unit_tests():
    """Exécute les tests unitaires avec pytest."""
    run_command([sys.executable, "-m", "pytest", "-q", TEST_FILE])


# --------------------------------------------------------------------------- #
# Étapes du pipeline ML
# --------------------------------------------------------------------------- #
def run_prepare(data_path="adult.zip"):
    """Prépare les données. Retourne X_train, X_test, y_train, y_test, scaler."""
    data = prepare_data(data_path)
    X_train, X_test, y_train, _, _ = data
    print(
        f"Données prêtes : train={X_train.shape}, test={X_test.shape}, "
        f"%>50K (train)={y_train.mean():.3f}"
    )
    return data


def run_train(X_train, y_train, model_type="gb", k=12, tune_k=False):
    """Entraîne le modèle (gb ou knn)."""
    if model_type == "knn" and tune_k:
        k = find_best_k(X_train, y_train)
    model = train_model(X_train, y_train, model_type, n_neighbors=k)
    print(f"Modèle entraîné : {type(model).__name__}")
    return model


def run_save(model, path="model.joblib", scaler=None):
    """Sauvegarde le modèle."""
    save_model(model, path, scaler=scaler)


def run_load(path="model.joblib"):
    """Charge le modèle sauvegardé."""
    model = load_model(path)
    print(f"Modèle chargé : {path}")
    return model


def run_evaluate(model, X_test, y_test):
    """Évalue le modèle et affiche l'importance des variables si disponible."""
    metrics = evaluate_model(model, X_test, y_test, name=type(model).__name__)
    if hasattr(model, "feature_importances_"):
        print("Top 10 des variables importantes :")
        print(feature_importance(model, X_test.columns, top=10).round(3))
    return metrics


def run_predict(model, X_test, y_test=None, n=5):
    """Prédit les n premières observations du jeu de test et les affiche."""
    result = predict_model(model, X_test.head(n))
    if y_test is not None:
        result["réel"] = ["<=50K" if v == 0 else ">50K" for v in y_test.head(n)]
    print("\n=== Prédictions ===")
    print(result)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Pipeline ML - Adult Income")
    p.add_argument(
        "--data",
        default="adult.zip",
        help="adult.zip, dossier adult.data/adult.test, ou CSV nettoyé",
    )
    p.add_argument("--model", default="model.joblib", help="Chemin du modèle")
    p.add_argument("--model-type", choices=["gb", "knn"], default="gb")
    p.add_argument("--k", type=int, default=12, help="k du KNN (défaut 12)")
    p.add_argument("--tune-k", action="store_true", help="k du KNN par CV")
    p.add_argument("--prepare", action="store_true", help="Préparer les données")
    p.add_argument("--train", action="store_true", help="Entraîner le modèle")
    p.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    p.add_argument("--predict", action="store_true", help="Prédire sur le test")
    p.add_argument("--save", action="store_true", help="Sauvegarder le modèle")
    p.add_argument("--all", action="store_true", help="Tout le pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    actions = [args.prepare, args.train, args.evaluate, args.predict, args.save]
    if not any(actions + [args.all]):
        print("Aucune action demandée. Utilisez --help.")
        return

    X_train, X_test, y_train, y_test, scaler = run_prepare(args.data)

    model = None
    if args.train or args.all or args.save:
        model = run_train(X_train, y_train, args.model_type, args.k, args.tune_k)

    if args.save or args.all:
        run_save(model, args.model, scaler=scaler)

    if args.evaluate or args.predict or args.all:
        if model is None:
            model = run_load(args.model)
        if args.evaluate or args.all:
            run_evaluate(model, X_test, y_test)
        if args.predict or args.all:
            run_predict(model, X_test, y_test)


if __name__ == "__main__":
    main()
