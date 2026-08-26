import numpy as np
class LogisticRegression:
    def __init__(self) -> None:
        pass
    
    def forward(self, X):
        # X: N,D design matrix
        self.X = X
        self.weights = np.random.randn(X.shape[1]) #D, 
        self.bias = np.zeros(1)
        logits = X @ self.weights + self.bias # N,
        self.probs = 1 / (1 + np.exp(-logits)) #N,
        return self.probs
    
    def backwards(self, delta): 
        self.dX = np.outer(delta, self.weights) #N,D
        self.dW = self.X.T @ delta #D,
        self.db = delta
    
    def update(self, eta):
        self.weights -= eta * self.dW
        self.bias -= eta * self.bias
        
    def IRLS(self, X, y, epochs):
        #Iteratively reweighted least squares
        N, D = X.shape[0], X.shape[1]
        self.weights = np.zeros(D)
        z = np.zeros(N)
        s = np.zeros(N)
        for _ in range(epochs):
            for n in range(N):
                a_n = self.weights.T @ X[n] #a_n = D, x D,
                mu_n = 1 / (1 + np.exp(-a_n)) # scalar
                s[n] = mu_n*(1 - mu_n) # scalar
                z[n] = a_n + (y[n] - mu_n) / s[n]
            S = np.diag(s) # N,N
            self.weights = np.linalg.solve(X.T @ S @ X, X.T @ S @ z)

        # Vectorized Version
        '''
        for _ in range(epochs):
            a = X @ self.weights #N,D x D, = N,
            mu = self.sigmoid(a) # N,
            s = mu * (1 - mu) # N, 
            z = a + (y - mu) / s
            S = np.diag(s)
            self.weights = np.linalg.solve(X.T @ S @ X, X.T @ S @ z)
        '''
        return 
   

    
        