import numpy as np
class KNN:
    def __init__(self, K, distance) -> None:
        self.K = K
        self.distance = distance
    
    def fit(self, X_train, y_train):
        # X: N, D (N examples with D dims)
        # y: N, labels
        self.X_train = X_train
        self.y_train = y_train
    
    def predict(self, X_test):
        # X_test: M,D
        distances = self.distance(X_test, self.X_train) # M, N
        top_K_idxs = np.argsort(distances, axis= -1)[:, :self.K] # M, K
        K_labels = self.y_train[top_K_idxs] 
        preds = []
        for label in K_labels:
            vals, counts = np.unique(label, return_counts=True)
            pred = vals[np.argmax(counts, axis= -1)]
            preds.append(pred)
        return np.array(preds)
        
        
        
        