"""API FastAPI : prédiction du risque de diabète (SP-28).

Les pipelines sont chargés depuis le Model Registry de MLflow (alias "production").
"""
import json
import logging
import os
import time
import warnings
from pathlib import Path

# L'API gère elle-même les nouveaux essais (load_models) : pas d'attente cachée côté MLflow
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

# Avertissement sans conséquence : l'imputer renvoie un tableau numpy
warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = logging.getLogger("uvicorn.error")

ROOT_PATH = Path(__file__).resolve().parent.parent
BUNDLE_PATH = ROOT_PATH / "models" / "bundle.joblib"

# Mêmes noms que dans src/mlflow_utils.py
RISK_MODEL_NAME = "diabetes_risk_pipeline"
CLUSTER_MODEL_NAME = "diabetes_cluster_pipeline"
PRODUCTION_ALIAS = "production"

# Docker : MLFLOW_TRACKING_URI=http://mlflow:5000. En local : la base SQLite écrite par src/train.py.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT_PATH / 'mlflow' / 'mlflow.db').as_posix()}"
)


def load_from_registry():
    """Charge les deux pipelines en production et les informations stockées avec le modèle."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(RISK_MODEL_NAME, PRODUCTION_ALIAS)
    return {
        "pipeline_risque": mlflow.sklearn.load_model(f"models:/{RISK_MODEL_NAME}@{PRODUCTION_ALIAS}"),
        "pipeline_cluster": mlflow.sklearn.load_model(f"models:/{CLUSTER_MODEL_NAME}@{PRODUCTION_ALIAS}"),
        "features": json.loads(version.tags["features"]),
        "zero_as_missing": json.loads(version.tags["zero_as_missing"]),
        "high_risk_clusters": json.loads(version.tags["high_risk_clusters"]),
        "source": f"mlflow:{RISK_MODEL_NAME}@{PRODUCTION_ALIAS} (version {version.version})",
    }


def load_models(retries=20, delay=3):
    """Registry MLflow d'abord. Si aucun modèle n'y est encore en production, bundle local."""
    for attempt in range(1, retries + 1):
        try:
            return load_from_registry()
        except MlflowException as error:
            if error.error_code == "RESOURCE_DOES_NOT_EXIST":
                logger.warning("Aucun modèle '%s' dans le registry MLflow.", PRODUCTION_ALIAS)
                break
            # Le serveur MLflow démarre peut-être encore : on réessaie
            logger.warning("MLflow injoignable (essai %s/%s) : %s", attempt, retries, error)
        except Exception as error:
            logger.warning("MLflow injoignable (essai %s/%s) : %s", attempt, retries, error)
        time.sleep(delay)

    logger.warning("Chargement du bundle local %s à la place du registry.", BUNDLE_PATH)
    models = joblib.load(BUNDLE_PATH)
    models["source"] = f"fichier local {BUNDLE_PATH.name}"
    return models


models = load_models()
logger.info("Modèle chargé depuis : %s", models["source"])

app = FastAPI(title="Diabetes Risk API", version="1.0")


class Patient(BaseModel):
    """Les 8 variables cliniques. Les champs vides (ou égaux à 0) pour
    Glucose, BloodPressure, SkinThickness, Insulin, BMI sont imputés."""

    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(..., gt=0, le=3)
    Age: int = Field(..., ge=1, le=120)


@app.get("/health")
def health():
    return {"status": "ok", "model_source": models["source"]}


@app.post("/reload")
def reload_models():
    """Recharge le modèle en production (à appeler après un réentraînement)."""
    global models
    models = load_models(retries=1, delay=0)
    return {"model_source": models["source"]}


@app.post("/predict")
def predict(patient: Patient):
    pipeline_risque = models["pipeline_risque"]
    pipeline_cluster = models["pipeline_cluster"]
    zero_as_missing = models["zero_as_missing"]

    # 1) Une ligne, colonnes dans le bon ordre
    row = pd.DataFrame([patient.model_dump()])[models["features"]].astype(float)

    # 2) 0 -> NaN pour les colonnes où 0 veut dire "valeur manquante"
    for col in zero_as_missing:
        if row.at[0, col] == 0:
            row.at[0, col] = np.nan

    # 3) Pas assez d'information : on refuse de prédire
    if row[zero_as_missing].isna().all(axis=1).iloc[0]:
        raise HTTPException(
            status_code=422,
            detail="Pas assez de données : renseignez au moins une mesure clinique.",
        )

    # 4) Prédictions
    risk = pipeline_risque.predict(row)[0]
    classes = list(pipeline_risque.named_steps["rf"].classes_)
    proba_high = float(pipeline_risque.predict_proba(row)[0][classes.index("risque élevé")])

    cluster = int(pipeline_cluster.predict(row)[0])
    cluster_risk = "risque élevé" if cluster in models["high_risk_clusters"] else "risque faible"

    return {
        "risk_category": str(risk),
        "probability_high_risk": round(proba_high, 3),
        "cluster": cluster,
        "cluster_risk": cluster_risk,
    }
