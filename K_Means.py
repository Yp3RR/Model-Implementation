import numpy as np


class ScratchKMeans:
    """
    K-Means Clustering via Expectation-Maximization loop.
    Assigns points to closest centroids, then updates centroid locations.
    """

    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        X = np.array(X)
        np.random.seed(42)  # Fixed seed for reproducible centroid initialization

        # Step 1: Initialize centroids randomly from dataset rows
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # Step 2: Assign Phase - Distance from every point to all K centroids
            distances = np.array([np.sqrt(np.sum((X - c) ** 2, axis=1)) for c in self.centroids])
            labels = np.argmin(distances, axis=0)

            # Step 3: Move Phase - Recalculate centroids as cluster means
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Convergence check: Stop early if centroids do not move
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = labels


if __name__ == "__main__":
    print("🚀 EXECUTING SCRATCH K-MEANS")

    # 2 distinct clusters in 2D space
    X_train = np.array([
        [1.0, 2.0], [1.5, 1.8], [2.0, 2.1],  # Cluster Group 1
        [8.0, 8.0], [8.5, 8.1], [9.0, 8.5]  # Cluster Group 2
    ])

    kmeans = ScratchKMeans(k=2)
    kmeans.fit(X_train)

    print(f"Cluster Assignments: {kmeans.labels_}")
    print(f"Learned Centroids:\n{kmeans.centroids}")