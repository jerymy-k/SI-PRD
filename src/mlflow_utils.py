"""MLflow helpers: experiment tracking, traceability and Model Registry."""
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import imblearn
import joblib
import matplotlib

matplotlib.use("Agg")  # draw images without a screen (scripts, Docker, Airflow)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import skops.io as sio
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Names used in the Model Registry. The API (app/main.py) loads these two models.
RISK_MODEL_NAME = "diabetes_risk_pipeline"
CLUSTER_MODEL_NAME = "diabetes_cluster_pipeline"
PRODUCTION_ALIAS = "production"
HIGH_RISK_LABEL = "risque élevé"


def setup_mlflow(tracking_uri, experiment_name, artifact_location=None):
    """Initializes the MLflow tracking URI and sets the experiment."""
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def build_traceability(features, data_path, df):
    """Collects what is needed to reproduce a run: features, library versions, dataset signature."""
    data_path = Path(data_path)
    return {
        "feature_cols": ", ".join(features),
        "n_features": len(features),
        "dataset_file": data_path.name,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "dataset_rows": df.shape[0],
        "dataset_cols": df.shape[1],
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "imblearn_version": imblearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "mlflow_version": mlflow.__version__,
    }


def _trusted_types(model):
    """Types MLflow must trust to reload this model (it is saved in the safe 'skops' format)."""
    return sio.get_untrusted_types(data=sio.dumps(model))


def _log_sklearn_model(model, name, input_example=None, registered_model_name=None):
    return mlflow.sklearn.log_model(
        model,
        name=name,
        input_example=input_example,
        registered_model_name=registered_model_name,
        skops_trusted_types=_trusted_types(model),
    )


def _log_classification_results(y_true, y_pred):
    """Logs accuracy, precision, recall, F1 and the confusion matrix image of the active run."""
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_risque_eleve": precision_score(y_true, y_pred, pos_label=HIGH_RISK_LABEL),
        "recall_risque_eleve": recall_score(y_true, y_pred, pos_label=HIGH_RISK_LABEL),
        "f1_risque_eleve": f1_score(y_true, y_pred, pos_label=HIGH_RISK_LABEL),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    })

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


def log_clustering_run(run_name, kmeans, scaler, silhouette, traceability):
    """Logs the K-Means run: k, inertia, silhouette, scaler parameters and both model files."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "n_clusters": kmeans.n_clusters,
            "init": kmeans.init,
            "n_init": kmeans.n_init,
            "random_state": kmeans.random_state,
        })
        mlflow.log_params({f"scaler_{key}": value for key, value in scaler.get_params().items()})
        mlflow.log_params(traceability)
        mlflow.log_metrics({"silhouette": silhouette, "inertia": kmeans.inertia_})

        # Fitted values of the scaler (mean and standard deviation per feature)
        mlflow.log_dict(
            {
                "features": list(scaler.feature_names_in_),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            "scaler_stats.json",
        )

        # K-Means and scaler files as artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            kmeans_path = Path(tmp_dir) / "kmeans.joblib"
            scaler_path = Path(tmp_dir) / "scaler.joblib"
            joblib.dump(kmeans, kmeans_path)
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(str(kmeans_path), artifact_path="clustering")
            mlflow.log_artifact(str(scaler_path), artifact_path="clustering")

        return mlflow.active_run().info.run_id


def log_classification_run(run_name, model, params, y_test, y_pred, traceability, extra_params=None):
    """Logs one classification model: hyperparameters, metrics, confusion matrix and the model."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_params(traceability)
        if extra_params:
            mlflow.log_params(extra_params)
        _log_classification_results(y_test, y_pred)
        _log_sklearn_model(model, name="model")
        return mlflow.active_run().info.run_id


def log_and_register_pipelines(
    run_name, bundle, best_params, param_grid, cv_score, X_test, y_test, traceability, extra_params=None
):
    """Logs the tuned model run, then registers the full pipelines and moves them to production.

    `bundle` is the dict built by src.pipeline.build_inference_pipelines.
    """
    pipeline_risque = bundle["pipeline_risque"]
    pipeline_cluster = bundle["pipeline_cluster"]
    input_example = X_test.head(5)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(best_params)
        mlflow.log_params(traceability)
        if extra_params:
            mlflow.log_params(extra_params)
        # Hyperparameters explored by the grid search
        mlflow.log_dict({key: [str(v) for v in values] for key, values in param_grid.items()}, "param_grid.json")
        mlflow.log_metric("best_cv_f1_risque_eleve", cv_score)
        _log_classification_results(y_test, pipeline_risque.predict(X_test))

        risk_info = _log_sklearn_model(
            pipeline_risque, "pipeline_risque", input_example, registered_model_name=RISK_MODEL_NAME
        )
        cluster_info = _log_sklearn_model(
            pipeline_cluster, "pipeline_cluster", input_example, registered_model_name=CLUSTER_MODEL_NAME
        )
        run_id = mlflow.active_run().info.run_id

    # Move both new versions to production, and store what the API needs next to the risk model
    client = MlflowClient()
    risk_version = risk_info.registered_model_version
    cluster_version = cluster_info.registered_model_version
    client.set_registered_model_alias(RISK_MODEL_NAME, PRODUCTION_ALIAS, risk_version)
    client.set_registered_model_alias(CLUSTER_MODEL_NAME, PRODUCTION_ALIAS, cluster_version)
    for key in ("features", "zero_as_missing", "high_risk_clusters"):
        client.set_model_version_tag(RISK_MODEL_NAME, risk_version, key, json.dumps(bundle[key]))
    client.set_model_version_tag(RISK_MODEL_NAME, risk_version, "cluster_model_version", str(cluster_version))

    return run_id, risk_version, cluster_version
