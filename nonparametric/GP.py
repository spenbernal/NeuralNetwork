import numpy as np
class GaussianProcess:
    def __init__(self, mercer_kernel, mean_func: np.vectorize, noise_var= 0.0) -> None:
        self.mercer_kernel = mercer_kernel
        self.mean_func = mean_func
        self.noise_var = noise_var
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # X: N, D
        # Y: N,
        self.X_train = X_train
        self.y_train = y_train
        
    # Noise free predictions
    def predict(self, X_test):
        # M, D
        mu_X = self.mean_func(self.X_train) # N,
        K_XX = self.mercer_kernel(self.X_train, self.X_train)
        mu_star = self.mean_func(X_test)
        K_X_star = self.mercer_kernel(self.X_train, X_test)
        K_star_star = self.mercer_kernel(X_test, X_test)
        mu = mu_star + K_X_star.T @ np.linalg.solve(K_XX, self.y_train - mu_X)
        Sigma = K_star_star - K_X_star.T @ np.linalg.solve(K_XX, K_X_star)
        
        preds = np.random.multivariate_normal(mean= mu, cov= Sigma, size= X_test.size)
        return preds
    
    # Noisy predictions
    def predict_(self, X_test):
        # M, D
        mu_X = self.mean_func(self.X_train)
        mu_star = self.mean_func(X_test)
        K_XX = self.mercer_kernel(self.X_train, self.X_train)
        K_X_star = self.mercer_kernel(self.X_train, X_test)
        K_star_star = self.mercer_kernel(X_test, X_test)
        K_sigma = K_XX + self.noise_var * np.eye(self.y_train.size)
        mu_posterior = mu_star + K_X_star.T @ np.linalg.solve(K_sigma, self.y_train - mu_X)
        Sigma_posterior = K_star_star - K_X_star.T @ np.linalg.solve(K_sigma, K_X_star)
        return mu_posterior, Sigma_posterior
        

class MercerKernels:
    def rbf(self, X1, X2, bandwidth= 1.0, variance= 1.0):
        # X1: N,D
        # X2: M,D
        diff = X1[:, None, :] - X2[None, :, :] # N, M, D
        squared_dist = np.linalg.norm(diff**2, axis= -1)
        return variance * np.exp(-0.5 * squared_dist / bandwidth**2)
        
        