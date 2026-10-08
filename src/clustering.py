import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_kmeans(df_scaled, n_clusters=2, random_state=42, n_init=10):
    """Trains a KMeans clustering model and returns the model, labels, and silhouette score."""
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state, n_init=n_init)
    labels = kmeans.fit_predict(df_scaled)
    score = silhouette_score(df_scaled, labels)
    return kmeans, labels, score

def is_high_risk(row):
    """Clinical threshold logic to determine if a cluster mean represents high risk."""
    return (row['Glucose'] > 126) and (row['BMI'] > 30) and (row['DiabetesPedigreeFunction'] > 0.5)

def assign_risk_categories(df, labels):
    """Assigns the 'risk_category' to each patient based on cluster characteristics."""
    df_out = df.copy()
    df_out['Cluster'] = labels
    
    # Calculate cluster means
    cluster_means = df_out.groupby('Cluster').mean()
    
    # Identify high risk clusters
    high_risk_clusters = cluster_means[cluster_means.apply(is_high_risk, axis=1)].index.tolist()
    
    # Map back to dataframe
    df_out['risk_category'] = df_out['Cluster'].apply(
        lambda x: 'risque élevé' if x in high_risk_clusters else 'risque faible'
    )
    
    return df_out, high_risk_clusters

