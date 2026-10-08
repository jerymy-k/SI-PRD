"""API FastAPI : prédiction du risque de diabète (SP-28)."""
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Avertissement sans conséquence : l'imputer renvoie un tableau numpy
warnings.filterwarnings("ignore", message="X does not have valid feature names")

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "models" / "bundle.joblib"
bundle = joblib.load(BUNDLE_PATH)

pipeline_risque = bundle["pipeline_risque"]
pipeline_cluster = bundle["pipeline_cluster"]
FEATURES = bundle["features"]
ZERO_AS_MISSING = bundle["zero_as_missing"]
HIGH_RISK_CLUSTERS = bundle["high_risk_clusters"]

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
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: Patient):
    # 1) Une ligne, colonnes dans le bon ordre
    row = pd.DataFrame([patient.model_dump()])[FEATURES].astype(float)

    # 2) 0 -> NaN pour les colonnes où 0 veut dire "valeur manquante"
    for col in ZERO_AS_MISSING:
        if row.at[0, col] == 0:
            row.at[0, col] = np.nan

    # 3) Pas assez d'information : on refuse de prédire
    if row[ZERO_AS_MISSING].isna().all(axis=1).iloc[0]:
        raise HTTPException(
            status_code=422,
            detail="Pas assez de données : renseignez au moins une mesure clinique.",
        )

    # 4) Prédictions
    risk = pipeline_risque.predict(row)[0]
    classes = list(pipeline_risque.named_steps["rf"].classes_)
    proba_high = float(pipeline_risque.predict_proba(row)[0][classes.index("risque élevé")])

    cluster = int(pipeline_cluster.predict(row)[0])
    cluster_risk = "risque élevé" if cluster in HIGH_RISK_CLUSTERS else "risque faible"

    return {
        "risk_category": str(risk),
        "probability_high_risk": round(proba_high, 3),
        "cluster": cluster,
        "cluster_risk": cluster_risk,
    }
