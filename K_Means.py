import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

np.random.seed(42)
X = np.random.rand(300, 5) * np.array([10.0, 500.0, 0.5, 100.0, 2.0])

kmeans_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(
        n_clusters=4, 
        init='k-means++',      # Mitigates local minima traps
        n_init=10,             # Runs algorithm 10 times with different seeds and picks best
        max_iter=300,          # Hard cap on loops to avoid endless execution
        random_state=42
    ))
])

kmeans_pipeline.fit(X)

cluster_labels = kmeans_pipeline.named_steps['kmeans'].labels_

X_scaled = kmeans_pipeline.named_steps['scaler'].transform(X)
score = silhouette_score(X_scaled, cluster_labels)

print(f"Training completed. Silhouette Score for K=4: {score:.4f}")