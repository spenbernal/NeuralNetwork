import numpy as np
import scipy.stats as st

class Kmeans:
    def __init__(self, K) -> None:
        self.K = K
    
    def fit(self, X, epochs):
        # X: N,D design matrix
        N, D = X.shape
        Z = np.zeros((N, self.K))
        M = np.zeros((self.K, D))
        I = np.eye(self.K)
        init = np.random.randint(N)
        M[0] = X[init]
        # initialize centroids
        for k in range(1, self.K):
            # sequentially choose next centroid
            diff = X[:, None, :] - M[None, :k, :]
            dist = np.sum(diff**2, axis= 2)
            min_dist = np.min(dist, axis= 1)
            probs = min_dist / np.sum(min_dist)
            
            # choose next centroid
            idx = np.random.choice(N, p=probs)
            M[k] = X[idx]
            
            
        for e in range(epochs):
            # calculate nearest centroid for every point
            diff = X[:, None, :] - M[None, :, :] # N, K, D
            dist = np.sum(diff**2, axis= 2) # N,K
            centers = np.argmin(dist, axis= 1) # N, 
            Z = I[centers] # N,K
            distortion = np.sum((X - Z @ M)**2)
            print(f'Epoch: {e+1} | Distorion: {distortion}')
            
            # update centroids
            new_M = np.zeros_like(M)
            for k in range(self.K):
                points = X[centers == k]
                if len(points) > 0:
                    new_M[k] = points.mean(axis= 0)
                else:
                    new_M[k] = M[k]
            
            M = new_M
        
        self.M = M
        self.Z = Z
        
        return