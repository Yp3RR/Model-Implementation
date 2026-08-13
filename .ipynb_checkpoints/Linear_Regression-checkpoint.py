import numpy as np

class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):

        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):

            y_predicted = np.dot(X,self.weights) + self.bias
            error = y_predicted-y
            dw = 1/n_samples*np.dot(X.T,error)
            db = 1/n_samples*np.sum(error)
             
            self.weights -= self.lr*dw 
            self.bias -= self.lr*db 
 
            if epoch % 100 == 0:
                loss = np.mean(error ** 2)
                print(f"Epoch {epoch:4d} | MSE: {loss:.4f}")

    def predict(self, X):

        return np.dot(X,self.weights)+self.bias


if __name__ == "__main__":
    # 1. Python reads the main body data arrays first.
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=float)
    y_train = np.array([5, 11, 17, 23, 29], dtype=float)

    regressor = LinearRegressionFromScratch(learning_rate=0.01, epochs=2000)
    
    regressor.fit(X_train, y_train)
    
    X_new = np.array([[11, 12]], dtype=float)
    print("Prediction:", regressor.predict(X_new)[0])