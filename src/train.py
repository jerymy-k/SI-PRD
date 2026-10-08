import os
from pathlib import Path
import warnings

# Suppress sklearn warnings about feature names inside pipelines if desired
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from src.preprocessing import load_data, clean_data, scale_data, NAN_COLUMNS
from src.clustering import train_kmeans, assign_risk_categories
from src.classification import prepare_splits, apply_smote, train_baseline_models, tune_random_forest
from src.pipeline import build_inference_pipelines, save_bundle
from src.mlflow_utils import setup_mlflow, log_clustering_run, log_classification_run, log_tuned_model_run, register_best_model

ROOT_PATH = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_PATH / "data" / "raw"
PROCESSED_DATA_PATH = ROOT_PATH / "data" / "processed"
MODELS_PATH = ROOT_PATH / "models"

def main():
    # Setup MLflow to save locally inside the mlflow folder
    # Use SQLite so it doesn't clutter the root directory
    mlflow_db_path = ROOT_PATH / "mlflow" / "mlflow.db"
    mlflow_artifacts_path = ROOT_PATH / "mlflow" / "mlartifacts"
    
    # We will just point the tracking URI to the sqlite db
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
    
    print("2. Clustering (K-Means)...")
    kmeans, labels, silhouette = train_kmeans(df_scaled, n_clusters=2)
    df_clustered, high_risk_clusters = assign_risk_categories(df_imputed, labels)
    df_clustered.to_csv(PROCESSED_DATA_PATH / "data_clustered.csv", index=False)
    
    # Log Clustering
    log_clustering_run(
        run_name="kmeans_clustering",
        params={"n_clusters": 2, "init": "k-means++", "random_state": 42},
        silhouette=silhouette,
        model=kmeans
    )
    
    print("3. Classification Preparation (SMOTE)...")
    X_train, X_test, y_train, y_test = prepare_splits(df_clustered, target_col='risk_category')
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    print("4. Training Baseline Models...")
    baseline_results = train_baseline_models(X_train_resampled, y_train_resampled, X_test, y_test)
    
    # Log Baselines
    for name, res in baseline_results.items():
        metrics = {
            'accuracy': res['classification_report']['accuracy'],
            'macro_f1': res['classification_report']['macro avg']['f1-score'],
            'risque_eleve_f1': res['classification_report']['risque élevé']['f1-score']
        }
        log_classification_run(
            run_name=name,
            params=res['model'].get_params(),
            metrics=metrics,
            model=res['model']
        )
    
    print("5. Hyperparameter Tuning (Random Forest)...")
    best_rf, best_params, best_cv_score = tune_random_forest(X_train, y_train)
    
    # Evaluate Tuned RF on Test Set
    preds_tuned = best_rf.predict(X_test)
    from sklearn.metrics import classification_report
    report_tuned = classification_report(y_test, preds_tuned, output_dict=True)
    
    metrics_tuned = {
        'cv_score': best_cv_score,
        'accuracy': report_tuned['accuracy'],
        'risque_eleve_f1': report_tuned['risque élevé']['f1-score']
    }
    
    # Log Tuned Model
    tuned_run_id = log_tuned_model_run(
        run_name="random_forest_tuned",
        params=best_params,
        metrics=metrics_tuned,
        best_rf=best_rf
    )
    
    # Register best model
    print("6. Registering Best Model to MLflow...")
    register_best_model(tuned_run_id, "diabetes_risk_random_forest")
    
    print("7. Assembling Pipelines and Saving Bundle...")
    os.makedirs(MODELS_PATH, exist_ok=True)
    pipelines = build_inference_pipelines(
        imputer, scaler, best_rf, kmeans,
        features=list(X_train.columns),
        zero_as_missing=NAN_COLUMNS,
        high_risk_clusters=high_risk_clusters,
    )
    save_bundle(pipelines, MODELS_PATH / "bundle.joblib")
    
    print("Pipeline execution complete!")

if __name__ == "__main__":
    main()
