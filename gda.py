import numpy as np

class GDA:
    def __init__(self, M, N) -> None:
        self.M = M
        self.N = N
        self.sigma = np.zeros((N, N))
    
    def train(self, X_train, y_train):
        idx_0 = np.where(y_train == 0)[0]
        idx_1 = np.where(y_train == 1)[0]
        
        self.phi = np.sum(y_train[idx_1]) / self.M
        
        self.mu0 = X_train[idx_0, :].mean(axis= 0)
        self.mu1 = X_train[idx_1, :].mean(axis= 0)
        
        self.sigma = ((X_train[idx_0, :] - self.mu0).T @  (X_train[idx_0, :] - self.mu0) 
        + (X_train[idx_1, :] - self.mu1).T @  (X_train[idx_1, :] - self.mu1)) / self.M
        
    def predict(self, X_test):
        inv_sigma = np.linalg.inv(self.sigma)
        X0 = X_test - self.mu0
        X1 = X_test - self.mu1
        const = 1/((2*np.pi)**(self.N/2) * np.sqrt(np.linalg.det(self.sigma)))
        q0 = np.sum((X0 @ inv_sigma) * X0, axis= 1)
        q1 = np.sum((X1 @ inv_sigma) * X1, axis= 1)
        
        p0 = const*np.exp((-1/2) * q0)
        p1 = const*np.exp((-1/2) * q1)
        
        probs = p1 * self.phi / (p0 * (1-self.phi) + p1*self.phi)
        prediction = (probs >= 0.5).astype(int)
        return prediction, probs
         
        
        