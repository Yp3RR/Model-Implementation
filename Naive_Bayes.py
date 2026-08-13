import numpy as np


class ScratchNaiveBayes:
    """
    Naive Bayes Classifier for binary feature vectors.
    Includes Laplace Smoothing (+1/+2) and Log-Sum computation.
    """

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.classes = np.unique(y)

        # Calculate Class Priors: P(C)
        self.priors = {c: np.mean(y == c) for c in self.classes}

        # Calculate Likelihoods: P(X_i | C) with Laplace Smoothing
        self.likelihoods = {}
        for c in self.classes:
            X_c = X[y == c]
            # (Feature Counts + 1) / (Total Class Rows + 2)
            self.likelihoods[c] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)

    def predict(self, X_new):
        X_new = np.array(X_new)
        predictions = []

        for point in X_new:
            class_scores = {}
            for c in self.classes:
                # Log-Sum Trick to prevent underflow
                log_prior = np.log(self.priors[c])

                # Retrieve probability array based on feature presence (1 or 0)
                probs = np.where(point == 1, self.likelihoods[c], 1 - self.likelihoods[c])
                log_likelihood = np.sum(np.log(probs))

                class_scores[c] = log_prior + log_likelihood

            # Argmax: Select class with maximum log posterior
            best_class = max(class_scores, key=class_scores.get)
            predictions.append(best_class)

        return np.array(predictions)


if __name__ == "__main__":
    print("🚀 EXECUTING SCRATCH NAIVE BAYES")

    # Binary features: e.g., contains ["Offer", "Free", "Meeting"]
    X_train = [[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 1], [0, 0, 1], [0, 1, 1]]
    y_train = [1, 1, 1, 0, 0, 0]  # 1 = Spam, 0 = Normal

    nb = ScratchNaiveBayes()
    nb.fit(X_train, y_train)

    X_test = [[1, 1, 0], [0, 0, 1]]
    predictions = nb.predict(X_test)

    print(f"Test Input:   {X_test}")
    print(f"Predictions:  {predictions} (Expected: [1, 0])")