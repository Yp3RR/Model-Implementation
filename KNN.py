import numpy as np


class ScratchKNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # Training is simply storing the dataset in memory
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X_new):
        X_new = np.array(X_new)
        predictions = []

        for point in X_new:
            # 1. Calculate Euclidean distance: sqrt(sum((x1 - x2)^2))
            distances = np.sqrt(np.sum((self.X_train - point) ** 2, axis=1))

            # 2. Get indices of the K smallest distances
            k_indices = np.argsort(distances)[:self.k]

            # 3. Retrieve their labels and pick the majority vote
            k_labels = self.y_train[k_indices]
            most_common = np.bincount(k_labels).argmax()
            predictions.append(most_common)

        return np.array(predictions)


if __name__ == "__main__":
    print("🚀 EXECUTING SCRATCH KNN")

    # 2D features: low coordinates vs high coordinates
    X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]
    y_train = [0, 0, 0, 1, 1, 1]

    knn = ScratchKNN(k=3)
    knn.fit(X_train, y_train)

    X_test = [[2, 2], [7, 6]]
    predictions = knn.predict(X_test)

    print(f"Test Input:   {X_test}")
    print(f"Predictions:  {predictions} (Expected: [0, 1])")