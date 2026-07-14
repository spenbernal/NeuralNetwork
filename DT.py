import numpy as np
class DecisionTree:
    def __init__(self) -> None:
        pass
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        # X_train: N,D 
        # y_train: N, 
        
        D = X_train.shape[1]
        C = np.unique(y_train)
        for d in range(D):
            # get thresholds at feature d
            thresholds = np.unique(X_train[:, d], sorted= True)
            best_gini = np.inf
            # find best splits on threshold for feature
            for t in thresholds:
                left_mask = (X_train[:, d] <= t)
                right_mask = (X_train[:, d] > t)
                left_split = X_train[left_mask]
                right_split = X_train[right_mask]
                left_count = left_split.size
                right_count = right_split.size
                total_count = left_count + right_count
                y_left = y_train[left_mask]
                y_right = y_train[right_mask]
                probs_l = [np.mean(y_left == c) for c in range(C)]
                probs_r = [np.mean(y_right == c) for c in range(C)]
                gini_left = 1 - np.sum(np.square(probs_l))
                gini_right = 1 - np.sum(np.square(probs_r))
                gini = (left_count / total_count) * gini_left + (right_count / total_count) * gini_right
                best_gini = np.minimum(gini, best_gini)
            
                
                    
                
                
                
                
            