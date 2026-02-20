import numpy as np

class Perceptron:
    def __init__(self, features) -> None:
        self.N = features
        self.weights = np.random.randn(self.N)
        self.bias = np.random.randn()
    
    def forward(self, X):
        return X @ self.weights + self.bias
    
    def train(self, X, Y, epochs, lr):
        for e in range(epochs):
            for x, y in zip(X, Y):
                z = self.forward(x)
                y_hat = 1 if z >= 0 else 0
                if y_hat != y:
                    self.weights += lr * (y - y_hat) * x
                    self.bias += lr * (y - y_hat)
        return 
                    
                
                
            