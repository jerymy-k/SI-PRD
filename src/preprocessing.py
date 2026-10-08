import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Columns where 0 means "missing value" (used for cleaning and saved in the bundle)
NAN_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data(filepath):
    """Loads the raw dataset, without the leftover CSV index column."""
    df = pd.read_csv(filepath)
    # "Unnamed: 0" is the row index saved by the CSV export, not a clinical variable
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def clean_data(df):
    """Replaces physiological zeros with NaN and imputes them using KNN."""
    df_clean = df.copy()
    nan_columns = NAN_COLUMNS
    
    # Replace 0s with NaN for clinical columns
    df_clean[nan_columns] = df_clean[nan_columns].replace(0, np.nan)
    
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5, weights="uniform", metric="nan_euclidean")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_clean), 
        columns=df_clean.columns
    )
    
    return df_imputed, imputer

def scale_data(df):
    """Scales data using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df), 
        columns=df.columns
    )
    return df_scaled, scaler

