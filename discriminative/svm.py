import numpy as np
# Linear SVM
class SVM:
    def __init__(self, N, C, kernel) -> None:
        self.N = N
        self.C = C
        self.weights = np.random.randn(N)
        self.bias = np.random.randn()

    def train(self, X, y, epochs, lr):
        for e in epochs:
            score = X @ self.weights + self.bias
            margin = y * score
            hinge = np.maximum(0, 1 - margin)
            loss = 0.5 * np.linalg.norm(self.weights)**2 + self.C * np.sum(hinge)
            
            violators = (margin < 1) 
            hinge_subgrad_w = self.C * X[violators].T @ y[violators] 
            regularizer = self.weights
            #update
            self.weights -= lr * (regularizer - hinge_subgrad_w)
            self.bias += lr * self.C * np.sum(y[violators])
            
            print(f'Epoch {e} | Loss: {loss}')
    
    def eval(self, X, y):
        score = X @ self.weights + self.bias
        margin = y * score
        hinge = np.maximum(0, 1 - margin)
        y_pred = np.where(score >= 0, 1, -1)
        acc = np.mean(y_pred == y)
        loss = 0.5 * np.linalg.norm(self.weights)**2 + self.C * np.sum(hinge)
        print(f'Loss: {loss:.5f} | Accuracy: {acc:.5f}')
        
        
        
        
        