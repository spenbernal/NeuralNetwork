import numpy as np
class KDE:
    def __init__(self, kernel) -> None:
        self.kernel = kernel 
        
    def fit(self, X_train):
        self.X_train = X_train
    
    def predict(self, X_test):
        # D,
        dist = np.linalg.norm(X_test - self.X_train, axis= -1)
        estimates = self.kernel(dist) # N,
        prediction = estimates.mean()
        return prediction
