import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(42)
X_raw = np.random.randn(100, 8) 

pca_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)) # Auto-selects components to preserve 95% variance
])

X_projected = pca_pipeline.fit_transform(X_raw)

pca_step = pca_pipeline.named_steps['pca']
num_components_chosen = X_projected.shape[1]
variance_per_component = pca_step.explained_variance_ratio_

print(f"Original Feature Count: {X_raw.shape[1]}")
print(f"Compressed Feature Count preserving 95% variance: {num_components_chosen}")
print(f"Information Retained per Component: {variance_per_component}")
print(f"Total Information Retained: {sum(variance_per_component) * 100:.2f}%")