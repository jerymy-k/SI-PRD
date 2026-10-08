# Code Restructuring Plan (Notebook to Modular Code)

Based on the analysis of `notebooks/main.ipynb` and the current project structure, here is the detailed plan to refactor the experimental notebook code into production-ready Python modules.

## 1. Analysis of `notebooks/main.ipynb`
The notebook currently contains the entire end-to-end Machine Learning lifecycle, logically broken down into several phases:
1.  **Data Preprocessing (Cells 1-30):** Loading data, handling zeros, KNN imputation, standard scaling, and saving datasets (`data_clean.csv`, `data_scaled.csv`).
2.  **Clustering (Cells 31-48):** KMeans clustering (k=2) to assign risk categories (`risk_category`) based on clinical thresholds, and saving `data_clustered.csv`.
3.  **Classification & SMOTE (Cells 49-56):** Splitting data (Train/Test), applying SMOTE for class imbalance, and training baseline models (Logistic Regression, Decision Tree, Random Forest).
4.  **Hyperparameter Tuning (Cells 57-61):** Using `GridSearchCV` to tune the Random Forest model and saving the tuned version.
5.  **Pipelines Assembly (Cells 62-63):** Building `pipeline_risk` and `pipeline_cluster` using scikit-learn pipelines and exporting them as a `bundle.joblib`.
6.  **MLflow Tracking (Cells 64-68):** Logging parameters, metrics, and models to MLflow, and registering the tuned Random Forest in the Model Registry.

## 2. Analysis of Project Folders
*   `src/`: Currently contains empty placeholder files (`preprocessing.py`, `clustering.py`, `classification.py`, `pipeline.py`, `mlflow_utils.py`). This is the target destination for our refactored code.
*   `mlflow/`: Contains `mlruns` placeholder. The actual local MLflow tracking data is currently saving to the root `mlartifacts/` and `mlflow.db`. 
*   `app/`, `airflow/`, `docker/`: Currently empty or placeholders, waiting for the core ML logic to be modularized first.

---

## 3. Proposed Restructuring Plan

We will extract the logic from the notebook and distribute it into the `src/` folder following clean code principles (modularity, reusability, separation of concerns). 

### A. `src/preprocessing.py`
**Responsibility:** Data loading, cleaning, and transformation.
*   `load_data(filepath)`: Loads the raw CSV.
*   `clean_data(df)`: Replaces physiological zeros with `NaN` and applies `KNNImputer`.
*   `scale_data(df)`: Applies `StandardScaler`.
*   *Outputs:* Cleaned and scaled DataFrames.

### B. `src/clustering.py`
**Responsibility:** Unsupervised learning and risk categorization.
*   `train_kmeans(df_scaled, n_clusters=2)`: Trains the KMeans model.
*   `assign_risk_categories(df, kmeans_model)`: Logic to calculate cluster means and apply the threshold rules (`Glucose > 126`, `BMI > 30`, etc.) to label patients as "risque élevé" or "risque faible".

### C. `src/classification.py`
**Responsibility:** Supervised learning, balancing, and tuning.
*   `prepare_splits(df)`: Creates X, y and performs `train_test_split`.
*   `apply_smote(X_train, y_train)`: Handles class imbalance.
*   `train_baseline_models(X_train, y_train)`: Trains Logistic Regression, Decision Tree, and Random Forest.
*   `tune_random_forest(X_train, y_train)`: Runs `GridSearchCV` to find the best Random Forest parameters.

### D. `src/pipeline.py`
**Responsibility:** Assembling the final inference pipelines for deployment.
*   `build_inference_pipelines(imputer, scaler, best_rf, kmeans)`: Assembles the scikit-learn `Pipeline` objects (`pipeline_risk` and `pipeline_cluster`).
*   `save_bundle(bundle_dict, filepath)`: Exports the pipelines using `joblib`.

### E. `src/mlflow_utils.py` & MLflow Folder Management
**Responsibility:** Centralizing MLflow tracking logic so it doesn't clutter the ML code.
*   `setup_mlflow(tracking_uri, experiment_name)`: Initializes the connection.
*   `log_clustering_run(...)`: Wrapper to log KMeans params, silhouette score, and model.
*   `log_classification_run(...)`: Wrapper to log model params, accuracy, F1-score, and the model artifact.
*   `register_best_model(...)`: Logic to tag and register the tuned model in the MLflow Model Registry.
*   *Folder structure tweak:* Ensure that the tracking URI points to a structured location (e.g., `sqlite:///mlflow/mlflow.db` and artifacts to `mlflow/mlartifacts/`) to keep the root directory clean, rather than dumping them directly in the root.

### F. `src/train.py` (New File)
**Responsibility:** The main execution script (Entrypoint).
*   This script will import all the functions defined above and run them sequentially, effectively replacing the "Run All" functionality of the Jupyter Notebook. It will be the script triggered by Airflow later.

---

## 4. Next Steps (Waiting for your approval)
If you approve this plan, I will:
1. Write the code for each module in `src/` as outlined above.
2. Update the MLflow tracking URI to keep logs neatly inside the `mlflow/` directory.
3. Ensure the original notebook remains untouched as a reference/exploration artifact.

**Please review this plan and let me know if I should proceed with writing the code, or if you'd like any adjustments!**

