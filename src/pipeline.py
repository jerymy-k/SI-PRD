import joblib
from sklearn.pipeline import Pipeline

def build_inference_pipelines(imputer, scaler, best_rf, kmeans):
    """
    Assembles the final scikit-learn pipelines for inference.
    """
    # 1. Pipeline for predicting Diabetes Risk Category
    pipeline_risk = Pipeline([
        ('imputer', imputer),
        ('classifier', best_rf)
    ])
    
    # 2. Pipeline for determining the Cluster
    pipeline_cluster = Pipeline([
        ('imputer', imputer),
        ('scaler', scaler),
        ('kmeans', kmeans)
    ])
    
    return {
        'pipeline_risk': pipeline_risk,
        'pipeline_cluster': pipeline_cluster
    }

def save_bundle(bundle_dict, filepath):
    """Saves the pipeline bundle to a joblib file."""
    joblib.dump(bundle_dict, filepath)

