import numpy as np
class GaussianProcess:
    def __init__(self, mercer_kernel, mean_func: np.vectorize) -> None:
        self.mercer_kernel = mercer_kernel
        self.mean_func = mean_func
        
    def fit(self, X_train, y_train):
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
        
        