import numpy as np
from abc import ABC, abstractmethod

class CART(ABC):
    def __init__(self, max_depth: int) -> None:
        self.j = None # feature
        self.t = None # threshold
        self.left = None # left subtree
        self.right = None # right subtree
        self.label = None # label if we reach a leaf
        self.max_depth = max_depth # maximum depth for stopping
        
    @abstractmethod
    def impurity(self, y: np.ndarray):
        pass
    
    @abstractmethod
    def leaf_label(self, y: np.ndarray):
        pass
    
    def stopping_conditions(self, y, depth):
        if self.impurity(y) == 0 or depth >= self.max_depth:
            return True
        return False
    
    def split_feature(self, X: np.ndarray, y: np.ndarray):
        # function that determines which feature to split on
        # returns feature, threshold, gini_index
        N, D = X.shape[0], X.shape[1]
        # (j,t) : score
        best_score = np.inf
        best_feature = None
        best_threshold = None
        for d in range(D):
            T = np.unique(X[:, d])
            for t in T:
                L = X[:, d] <= t
                R = ~L
                # check if we do not actually split the parent
                if L.sum() == 0 or R.sum() == 0:
                    continue 
        
                n_left = L.sum()
                n_right = R.sum()
                
                # total score
                score = (n_left/ N) * self.impurity(y[L]) + (n_right / N) * self.impurity(y[R]) 
                if score < best_score:
                    best_score = score
                    best_feature = d
                    best_threshold = t
        
        return best_feature, best_threshold # return feature, threshold
    
    def fit(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        
        if self.stopping_conditions(y, depth):
            self.label = self.leaf_label(y)
            return 
            
        # get best feature, threshold pair for node i
        feature, threshold = self.split_feature(X, y)
        
        # if no valid split is found
        if feature is None:
            self.label = self.leaf_label(y)
            return 
        
        self.j = feature
        self.t = threshold
        
        left = X[:, feature] <= threshold
        right = ~left
        
        self.left = self.new_child()
        self.right = self.new_child()
        
        self.left.fit(X[left], y[left], depth + 1)
        self.right.fit(X[right], y[right], depth + 1)
        return
    
    def new_child(self):
        return type(self)(max_depth= self.max_depth)
    
    def predict_one_sample(self, x):
        # x: D,
        if self.label is not None:
            return self.label

        assert self.left is not None
        assert self.right is not None
        
        if x[self.j] <= self.t:
            return self.left.predict_one_sample(x)
        else:
            return self.right.predict_one_sample(x)
            
    def predict(self, X):
        # X: M, D
        return np.array([self.predict_one_sample(x) for x in X])
class ClassificationTree(CART):
    def impurity(self, y: np.ndarray):
        _, counts = np.unique(y, return_counts= True)
        emp_dist = counts / len(y)
        gini = 1 - np.square(emp_dist).sum()
        return gini
    
    def leaf_label(self, y: np.ndarray):
        classes, counts = np.unique(y, return_counts= True)
        return classes[np.argmax(counts)]
    
class RegressionTree(CART):
    def impurity(self, y: np.ndarray):
        return np.mean((y - np.mean(y))**2)
    
    def leaf_label(self, y: np.ndarray):
        return np.mean(y)        

class Bagging:
    def __init__(self, M, Treeclass, **tree_kwargs) -> None:
        self.M = M
        self.Treeclass = Treeclass
        self.forest = [Treeclass(**tree_kwargs) for _ in range(M)]
        
    def fit(self, X, y):
        N = X.shape[0]
        for tree in self.forest:
            rand_samples = np.random.randint(low= 0, high= N, size= N)
            X_rand, y_rand = X[rand_samples], y[rand_samples]
            tree.fit(X_rand, y_rand)
            
        return

    def predict(self, X):
        preds = np.zeros((X.shape[0], self.M))
        for idx, tree in enumerate(self.forest):
            pred = tree.predict(X)
            preds[:, idx] = pred
            
        if issubclass(self.Treeclass, RegressionTree):
            return preds.mean(axis= -1)
        
        final_preds = np.empty(X.shape[0], dtype= preds.dtype)
        for idx, tree_preds in enumerate(preds):
            classes, counts = np.unique(tree_preds, return_counts= True)
            final_preds[idx] = classes[np.argmax(counts)]
        
        return final_preds
                
                
            
        