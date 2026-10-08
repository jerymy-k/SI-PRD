import mlflow
import mlflow.sklearn

def setup_mlflow(tracking_uri, experiment_name, artifact_location=None):
    """Initializes the MLflow tracking URI and sets the experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

def log_clustering_run(run_name, params, silhouette, model):
    """Logs the KMeans clustering run."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metric("silhouette", silhouette)
        mlflow.sklearn.log_model(model, "kmeans_model")

def log_classification_run(run_name, params, metrics, model):
    """Logs a classification model's run (parameters and performance metrics)."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        # Log tree based models with a trusted type to avoid warnings
        mlflow.sklearn.log_model(model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        return mlflow.active_run().info.run_id

def log_tuned_model_run(run_name, params, metrics, best_rf):
    """Logs the hyperparameter tuning result and saves the best model."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("data_version", "v1.0")
        mlflow.sklearn.log_model(best_rf, "random_forest_tuned_model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        return mlflow.active_run().info.run_id

def register_best_model(run_id, model_name, alias="version"):
    """Registers the model inside the MLflow Model Registry."""
    result = mlflow.register_model(f"runs:/{run_id}/random_forest_tuned_model", model_name)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(model_name, alias, result.version)
    return result
