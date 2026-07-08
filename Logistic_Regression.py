import numpy as np

class LogisticRegression:
    def __init__(self,epoch=1000,learning_rate=0.01):
        self.weights=None
        self.bias=None
        self.epochs = epoch
        self.lr = learning_rate

    def sigmoid(self,z):
        z = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z))

    def fit(self,X,y):
        n_samples,n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            z = np.dot(X,self.weights) + self.bias
            y_predicted = self.sigmoid(z)

            error = y_predicted-y
            dw = 1/n_samples * np.dot(X.T,error)
            db = 1/n_samples * np.sum(error)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(y_predicted + 1e-8) + 
                   (1-y) * np.log(1 - y_predicted + 1e-8))
                print(f"Epoch {epoch} | BCE Loss: {loss:.4f}")

    def predict_proba(self,X):

        z = np.dot(X,self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self,X):
        probablities = self.predict_proba(X)
        return np.array([1 if p>=0.5 else 0 for p in probablities])

if __name__ == "__main__":
    X_train = np.array([[1, 2], [2, 1], [8, 9], [9, 8], [1, 1]], dtype=float)
    y_train = np.array([0, 0, 1, 1, 0], dtype=float)
    
    clf = LogisticRegression(learning_rate=0.1, epoch=1000)
    clf.fit(X_train, y_train)
    
    X_test = np.array([[10, 10], [0, 1]], dtype=float)
    print("\n--- Model Inference Verification ---")
    print("Probabilities for test set: ", clf.predict_proba(X_test))
    print("Discrete Class Predictions:  ", clf.predict(X_test))
            
        