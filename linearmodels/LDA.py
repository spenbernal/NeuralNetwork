import numpy as np
import scipy.stats as st
# Generative classifier
class GDA:
    def __init__(self) -> None:
        pass
    
    def fit(self, X, y):
        # X: N,D design matrix
        # y: N, target vector (C classes)
        N, D = X.shape[0], X.shape[1]
        self.C = np.unique(y)
        self.mu = np.zeros((self.C,D))
        self.priors = np.zeros(self.C)
        # Untied covarainces
        self.Sigma = np.zeros((self.C, D, D))
        # Sigma = np.zeros(D,D) tied covariances
        # Sigma = np.zeros(C,D) diagonal covariances
        for c in range(self.C):
            X_c = X[y == c]
            N_c = X_c.shape[0]
            self.priors[c] = N_c / N
            self.mu[c] = X_c.mean(axis= 0)
            X_centered = X_c - self.mu[c]
            self.Sigma[c] = X_centered.T @ X_centered / N_c
            # Sigma += X_centered.T @ X_centered # for tied covariance implementation
            # Sigma[c] += np.sum(X_centered**2, axis= 0) / N_c # for diagonal covariances
        # Sigma /= N # for tied covariances
        
    def predict(self, X):
        # X: M,D design matrix
        M = X.shape[0] 
        log_priors = np.log(self.priors)
        scores = np.zeros((M, self.C))
        for c in range(self.C):
            log_mvn_pdf = st.multivariate_normal.logpdf(X, mean= self.mu[c], cov= self.Sigma[c])
            scores[:, c] = log_priors[c] + log_mvn_pdf
        preds = np.argmax(scores, axis= 1)
        return preds
    
class GaussianNB:
    def __init__(self) -> None:
        pass
    
    def fit(self, X, y):
        # X: (N,D) design matrix
        # y: (N,) target vector
        # assume continuous features
        self.C = np.unique(y)
        self.K = len(self.C)
        N = X.shape[0]
        D = X.shape[1]
        self.mu = np.zeros((D, self.K))
        self.var = np.zeros((D, self.K))
        self.priors = np.zeros(self.K)
        
        for k, c in enumerate(self.C):
            X_c = X[y == c]
            self.priors[k] = X_c.shape[0] / N
            for d in range(D):
                mu_dc = X_c[:, d].mean()
                self.mu[d, k] = mu_dc
                self.var[d, k] = np.mean((X_c[:, d] - mu_dc)**2)
                
        return
    
    def predict(self, X):
        M = X.shape[0]
        scores = np.zeros((M, self.K))
        log_prior = np.log(self.priors)
        for k in range(self.K):
            mu = self.mu[:, k]
            var = self.var[:, k]
            log_pdf = st.norm.logpdf(X, loc= mu, scale= np.sqrt(var))
            scores[:, k] = log_prior[k] + log_pdf.sum(axis= 1)
        
        return self.C[np.argmax(scores, axis= 1)]
        
