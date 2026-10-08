import joblib
from sklearn.pipeline import Pipeline

def build_inference_pipelines(imputer, scaler, best_rf, kmeans, features, zero_as_missing, high_risk_clusters):
    """
    Assembles the final scikit-learn pipelines for inference.
    The returned dict is the bundle loaded by the API (app/main.py): keep its keys in sync.
    """
    # 1. Pipeline for predicting Diabetes Risk Category
    pipeline_risque = Pipeline([
        ('imputer', imputer),
        ('rf', best_rf)
    ])
    
    # 2. Pipeline for determining the Cluster
    pipeline_cluster = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('kmeans', kmeans)
    ])
    
    return {
        'pipeline_risque': pipeline_risque,
        'pipeline_cluster': pipeline_cluster,
        'features': list(features),
        'zero_as_missing': list(zero_as_missing),
        'high_risk_clusters': list(high_risk_clusters),
    }

def save_bundle(bundle_dict, filepath):
    """Saves the pipeline bundle to a joblib file."""
    joblib.dump(bundle_dict, filepath)

