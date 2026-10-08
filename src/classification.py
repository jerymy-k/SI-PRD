from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def prepare_splits(df, target_col='risk_category', test_size=0.2, random_state=42):
    """Splits the dataframe into X (features) and y (target), dropping Cluster."""
    X = df.drop(columns=[target_col, 'Cluster'])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def apply_smote(X_train, y_train, random_state=42):
    """Applies SMOTE to balance the training set."""
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)

def train_baseline_models(X_train, y_train, X_test, y_test, random_state=42):
    """Trains and compares Logistic Regression, Decision Tree, and Random Forest."""
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            'model': model,
            'classification_report': classification_report(y_test, preds, output_dict=True)
        }
    return results

def tune_random_forest(X_train, y_train, random_state=42):
    """Tunes a Random Forest using GridSearchCV inside an imblearn pipeline to prevent data leakage."""
    pipeline_rf = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
        ('rf', RandomForestClassifier(random_state=random_state))
    ])
    
    param_grid = {
        'rf__n_estimators': [100, 200, 350],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
    }
    
    scorer = make_scorer(f1_score, pos_label='risque élevé')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(
        estimator=pipeline_rf,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Extract just the tuned Random Forest step (not the SMOTE part)
    best_rf = grid_search.best_estimator_.named_steps['rf']
    
    return best_rf, grid_search.best_params_, grid_search.best_score_

