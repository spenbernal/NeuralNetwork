import numpy as np

class LinearRegression:
    # Standard Linear Regression by OLS
    def __init__(self) -> None:
        pass
    
    def fit(self, X, y):
        # X: (N,D+1) design matrix
        # y: (N,) target vector
        self.weights = np.linalg.solve(X.T @ X, X.T @ y)
    
    def eval(self, X, y):
        # (M, D+1) design matrix
        y_pred = X @ self.weights
        mse = ((y - y_pred)**2).mean()
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        R2_score = 1 - ss_res / ss_tot
        return mse, R2_score
        
class RidgeRegression:
    # Linear regression with L2 regularization
    def __init__(self, lam) -> None:
        # include offset into weights (w0), so prepend X with a 1 for every row
        self.lam = lam # regularizer coefficient
    def fit(self, X, y, mode='SVD'):
        '''
        X: (N,D+1) matrix
        y: (N,) vector
        SVD approach
        '''
        if mode == 'SVD':
            # SVD approach
            U, S, V_t = np.linalg.svd(X, full_matrices=False) 
            R = np.matmul(U, np.diag(S)) 
            I = U.T @ U # N x N
            self.weights = V_t.T @(np.linalg.solve(R.T @ R + self.lam * I, R.T @ y))
        else:
            #QR appraoch
            D = X.shape[1]
            X_tilde = np.concatenate((X, np.sqrt(self.lam) * np.eye(D)), axis= 0)
            y_tilde = np.concatenate((y, np.zeros(D)), axis= 0)
            Q, R = np.linalg.qr(X_tilde)
            self.weights = np.linalg.solve(R, Q.T @ y_tilde)
        
    def eval(self, X, y):
        y_pred = np.matmul(X,self.weights)
        mse = ((y - y_pred)**2).mean()
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        R2_score = 1 - ss_res / ss_tot
        return mse, R2_score
class LassoRegression: 
    #Linear Regression with L1 regularization
    def __init__(self, lam) -> None:
        self.lam = lam
    def fit(self, X, y, epochs= 10):
        # soft thresholding
        # X: (N,D+1)
        # y: (N,)
        D = X.shape[1]
        self.weights = np.linalg.solve(X.T @ X + self.lam * np.eye(D), X.T @ y)
        
        mask = [True for _ in range(D)]
        for _ in range(epochs):
            for d in range(D):
                mask[d] = False
                a_d = X[:,d].T @ X[:,d] 
                residual = y - X[:, mask] @ self.weights[mask] #(N,)
                c_d = X[:,d].T @ residual # scalar
                if d == 0:
                    self.weights[d] = c_d / a_d
                else:
                    self.weights[d] = SoftThreshold(c_d / a_d, self.lam / a_d)
                mask[d] = True
    def eval(self, X, y):
        y_pred = np.matmul(X,self.weights)
        mse = np.mean((y - y_pred)**2)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        R2_score = 1 - ss_res / ss_tot
        return mse, R2_score
        
    
def SoftThreshold(x, delta):
    return np.sign(x) * np.maximum(0, (np.abs(x) - delta))
        
def CrossValidation(regularizers, X, y, K, model_name):
        '''
        X: training data (N,D+1)
        y: training labels (N,)
        regularizers: set of lambdas (L,)
        K: Number of cross folds (int)
        
        Hyperparameter testing for best lambda for lasso/ridge/elatic net
        '''
        #splits X,y into K num of subarrays
        train_sets_X = np.array_split(X, K, axis= 0)
        train_sets_y = np.array_split(y, K, axis= 0)
            
        loss_dict = {}
        for lam in regularizers:
            # initiate loss and model
            loss = 0
            if model_name == 'Ridge':
                model = RidgeRegression(lam)
            elif model_name == 'Lasso':
                model = LassoRegression(lam)
            #K fold cross validation
            for k in range(K):
                # get validation sets
                X_valid = train_sets_X[k]
                y_valid = train_sets_y[k]
                # get training sets
                X_train = np.concatenate([train_sets_X[i] for i in range(K) if i != k], axis= 0)
                y_train = np.concatenate([train_sets_y[i] for i in range(K) if i != k], axis= 0)
                
                # fit on training set
                model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = np.matmul(X_valid, model.weights)
                # MSE for fold k
                loss_k = np.mean((y_pred - y_valid)**2)
                loss += loss_k
                
            # average loss across K folds and store it in corresponding regularizer value
            avg_loss = 1/K * loss
            loss_dict[lam] = avg_loss
        
        best_lam = min(loss_dict, key=loss_dict.get) # type: ignore
        return best_lam
        
            
        

        
        
            
    
        