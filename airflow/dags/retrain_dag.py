"""DAG de réentraînement : extract -> preprocess -> retrain -> update registry."""
import os
import sys
from datetime import datetime

import requests
from airflow.sdk import dag, task  # Airflow 2 : from airflow.decorators import dag, task

# Dossier du projet monté dans le conteneur Airflow
PROJECT_DIR = os.getenv("PROJECT_DIR", "/opt/airflow/project")
sys.path.insert(0, PROJECT_DIR)

RAW_DATA = f"{PROJECT_DIR}/data/raw/data.csv"
PROCESSED_DIR = f"{PROJECT_DIR}/data/processed"
API_URL = os.getenv("API_URL", "http://api:8000")


@dag(
    dag_id="retrain_diabetes_model",
    start_date=datetime(2026, 10, 1),
    schedule="@weekly",
    catchup=False,
    tags=["si-prd"],
)
def retrain_diabetes_model():

    @task
    def extract():
        """Vérifie que les données brutes sont là et lisibles."""
        from src.preprocessing import load_data

        df = load_data(RAW_DATA)
        if df.empty:
            raise ValueError("data.csv est vide")
        print(f"{df.shape[0]} lignes, {df.shape[1]} colonnes")
        return RAW_DATA

    @task
    def preprocess(raw_path: str):
        """Nettoie (KNN) et standardise, puis sauvegarde les CSV."""
        from src.preprocessing import load_data, clean_data, scale_data

        df_imputed, _ = clean_data(load_data(raw_path))
        df_scaled, _ = scale_data(df_imputed)
        df_imputed.to_csv(f"{PROCESSED_DIR}/data_clean.csv", index=False)
        df_scaled.to_csv(f"{PROCESSED_DIR}/data_scaled.csv", index=False)
        return PROCESSED_DIR

    @task
    def retrain_and_register(_processed_dir: str):
        """Clustering, classifieurs, tuning, puis mise à jour du Model Registry."""
        from src.train import main

        main()

    @task
    def reload_api():
        """Demande à l'API de charger le nouveau modèle en production."""
        response = requests.post(f"{API_URL}/reload", timeout=120)
        response.raise_for_status()
        print(response.json())

    retrain_and_register(preprocess(extract())) >> reload_api()


retrain_diabetes_model()