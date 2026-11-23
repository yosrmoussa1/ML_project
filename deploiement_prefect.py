"""
deploiement_prefect.py
----------------------
Déploiements planifiés Prefect (le "worker" du Terminal 2).

Prérequis (Terminal 1) : prefect server start
Puis (Terminal 2) :
    prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
    python deploiement_prefect.py
Déclenchement manuel (Terminal 3) :
    prefect deployment ls
    prefect deployment run 'all/ml-pipeline-all'
    prefect deployment run 'train/ml-pipeline-train'
"""

from prefect import serve

from pipeline_prefect import (
    all_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    train_flow,
)

if __name__ == "__main__":
    serve(
        # Pipeline complet : tous les jours à 02:00
        all_flow.to_deployment(
            name="ml-pipeline-all",
            cron="0 2 * * *",
            tags=["full-pipeline", "mlops"],
            parameters={"skip_install": True},
        ),
        # Évaluation du modèle sauvegardé : tous les jours à 03:00
        evaluate_flow.to_deployment(
            name="ml-pipeline-evaluate",
            cron="0 3 * * *",
            tags=["evaluation", "mlops"],
        ),
        # Déploiements à lancer manuellement
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        code_flow.to_deployment(
            name="ml-pipeline-code",
            tags=["code-quality", "mlops"],
            parameters={"skip_install": True},
        ),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
    )
