import json
import shutil

import mlflow

MODEL_DIR = "model"


def find_latest_model():
    """Renvoie le dernier run terminé qui contient un modèle, et le chemin du modèle."""
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    for run in runs:
        # MLflow ajoute ce tag à chaque run où un modèle a été enregistré
        history = run.data.tags.get("mlflow.log-model.history")
        if history:
            artifact_path = json.loads(history)[-1]["artifact_path"]
            return run, artifact_path
    raise SystemExit("Aucun run terminé avec un modèle trouvé sur le serveur MLflow.")


if __name__ == "__main__":
    run, artifact_path = find_latest_model()
    print(f"Dernier modèle : run '{run.info.run_name}', params {run.data.params}")
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id, artifact_path=artifact_path, dst_path="."
    )
    shutil.move(local_path, MODEL_DIR)
    print(f"Modèle téléchargé dans ./{MODEL_DIR}")