import os
from pathlib import Path
import warnings

# Suppress sklearn warnings about feature names inside pipelines if desired
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.preprocessing import load_data, clean_data, scale_data, NAN_COLUMNS
from src.clustering import train_kmeans, assign_risk_categories
from src.classification import prepare_splits, apply_smote, train_baseline_models, tune_random_forest, RF_PARAM_GRID
from src.pipeline import build_inference_pipelines, save_bundle
from src.mlflow_utils import (
    setup_mlflow,
    build_traceability,
    log_clustering_run,
    log_classification_run,
    log_and_register_pipelines,
)

ROOT_PATH = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_PATH / "data" / "raw"
PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed"
MODELS_PATH = ROOT_PATH / "models"

def main():
    # If MLFLOW_TRACKING_URI is set (e.g. http://mlflow:5000 inside Docker), log to that server.
    # Otherwise save locally inside the mlflow folder, in a SQLite database.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        # The server decides where artifacts are stored
        setup_mlflow(tracking_uri, "Diabetes Risk Prediction")
    else:
        mlflow_db_path = ROOT_PATH / "mlflow" / "mlflow.db"
        mlflow_artifacts_path = ROOT_PATH / "mlflow" / "mlartifacts"
        tracking_uri = f"sqlite:///{mlflow_db_path.as_posix()}"
        artifact_uri = mlflow_artifacts_path.as_uri()
        setup_mlflow(tracking_uri, "Diabetes Risk Prediction", artifact_location=artifact_uri)

    print("1. Data Preprocessing...")
    df_raw = load_data(RAW_DATA_PATH / "data.csv")
    df_imputed, imputer = clean_data(df_raw)
    df_scaled, scaler = scale_data(df_imputed)
    
    # Save datasets
    df_imputed.to_csv(PROCESSED_DATA_PATH / "data_clean.csv", index=False)
    df_scaled.to_csv(PROCESSED_DATA_PATH / "data_scaled.csv", index=False)
    
    # Features, library versions and dataset signature, logged on every run
    traceability = build_traceability(list(df_imputed.columns), RAW_DATA_PATH / "data.csv", df_raw)
    
    print("2. Clustering (K-Means)...")
    kmeans, labels, silhouette = train_kmeans(df_scaled, n_clusters=2)
    df_clustered, high_risk_clusters = assign_risk_categories(df_imputed, labels)
    df_clustered.to_csv(PROCESSED_DATA_PATH / "data_clustered.csv", index=False)
    
    # Log Clustering
    log_clustering_run(
        run_name="kmeans_clustering",
        kmeans=kmeans,
        scaler=scaler,
        silhouette=silhouette,
        traceability=traceability
    )
    
    print("3. Classification Preparation (SMOTE)...")
    X_train, X_test, y_train, y_test = prepare_splits(df_clustered, target_col='risk_category')
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    split_info = {"train_rows": X_train.shape[0], "test_rows": X_test.shape[0], "resampling": "SMOTE"}
    
    print("4. Training Baseline Models...")
    baseline_results = train_baseline_models(X_train_resampled, y_train_resampled, X_test, y_test)
    
    # Log Baselines
    for name, res in baseline_results.items():
        log_classification_run(
            run_name=name,
            model=res['model'],
            params=res['model'].get_params(),
            y_test=y_test,
            y_pred=res['model'].predict(X_test),
            traceability=traceability,
            extra_params=split_info
        )
    
    print("5. Hyperparameter Tuning (Random Forest)...")
    best_rf, best_params, best_cv_score = tune_random_forest(X_train, y_train)
    
    print("6. Assembling Pipelines and Saving Bundle...")
    os.makedirs(MODELS_PATH, exist_ok=True)
    pipelines = build_inference_pipelines(
        imputer, scaler, best_rf, kmeans,
        features=list(X_train.columns),
        zero_as_missing=NAN_COLUMNS,
        high_risk_clusters=high_risk_clusters,
    )
    save_bundle(pipelines, MODELS_PATH / "bundle.joblib")
    
    print("7. Logging the Tuned Model and Registering the Pipelines in MLflow...")
    run_id, risk_version, cluster_version = log_and_register_pipelines(
        run_name="random_forest_tuned",
        bundle=pipelines,
        best_params=best_params,
        param_grid=RF_PARAM_GRID,
        cv_score=best_cv_score,
        X_test=X_test,
        y_test=y_test,
        traceability=traceability,
        extra_params=split_info
    )
    print(f"Production: risk pipeline v{risk_version}, cluster pipeline v{cluster_version} (run {run_id})")
    
    print("Pipeline execution complete!")

if __name__ == "__main__":
    main()
