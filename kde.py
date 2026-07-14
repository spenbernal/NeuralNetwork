import numpy as np
class KDE:
    def __init__(self, kernel, h) -> None:
        self.h = h
        self.kernel = kernel 
        
    def fit(self, X_train):
        self.X_train = X_train
    
    def predict(self, X_test):
        # D,
        D = X_test.size
        dist = np.linalg.norm(X_test - self.X_train, axis= -1)
        estimates = self.kernel(dist / self.h) # N,
        prediction = estimates.mean() / self.h**D
        return prediction
