import numpy as np

class PCA:
    def __init__(self, L) -> None:
        # Latent Dimension
        self.L = L
    
    def fit(self, X):
        # assume X is centered
        N = X.shape[0]
        # SVD decomposition
        U, S, Vh = np.linalg.svd(X)
        # store all val, vec pairs for visual analysis
        self.all_vals = S**2 / N
        self.all_vecs = Vh
        # The latent dims
        self.eig_vals = self.all_vals[:self.L]
        self.eig_vecs = Vh[:self.L, :]
            
        # latent representation
        Z = U[:, :self.L] * S[:self.L]
        
        return Z
    
    def embed(self, X_test):
        # project unseen data into training latent space
        return X_test @ self.eig_vecs.T
        
        
class PPCA:
    # Implementation of Linear Gaussian PPCA
    def fit(self, L, X):
        # L: latent dimension
        # X: N,D design matrix
        N, D = X.shape[0], X.shape[1]
        # empirical mean
        self.mu = np.mean(X, axis= 0)
        # centering matrix
        X_c = X - self.mu
        # empirical cov
        S = 1/N * X_c.T @ X_c
        # EVD of empirical cov
        eigvals, eigvecs = np.linalg.eigh(S)
        
        indices = np.argsort(eigvals)[::-1]
        # grab latent dimensions
        L_L = np.diag(eigvals[indices[:L]])
        U_L = eigvecs[:, indices[:L]]
        
        # arbitrary rotation matrix
        R = np.eye(L)
        # MLE for variance
        if L != D:
            self.sigma_squared = (1/(D-L)) * eigvals[indices[L:]].sum() 
        else:
            self.sigma_squared = 0
            
        # Weight matrix
        self.W = U_L @ np.sqrt((L_L - self.sigma_squared * np.eye(L))) @ R
        
        self.M = self.W.T @ self.W + self.sigma_squared * np.eye(L)
        
        posterior_mean = np.linalg.solve(self.M, self.W.T @ X_c.T).T
        posterior_var = self.sigma_squared * np.linalg.solve(self.M, np.eye(L))
        
        return posterior_mean, posterior_var

            
        
        
        