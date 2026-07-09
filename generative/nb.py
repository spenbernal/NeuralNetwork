import numpy as np
# binary classification
class NaiveBayes:
    def fit(self, X, y):
        m, n = X.shape[0], X.shape[1]
        self.y_0 = np.mean(y == 0)
        self.y_1 = np.mean(y == 1)
        self.mean = np.zeros((2,n))
        mask_0 = (y==0)
        mask_1 = (y==1)
        
        self.mean[0, :] = X[mask_0].mean(axis= 0)
        self.mean[1, :] = X[mask_1].mean(axis= 0)
    
    def predict(self, X, y):
        log_prob_0 = np.log(self.y_0) + np.sum(X * np.log(self.mean[0, :]) + (1-X)*np.log(1 - self.mean[0, :]), axis= 1)
        log_prob_1 = np.log(self.y_1) + np.sum(X * np.log(self.mean[1, :]) + (1-X)*np.log(1 - self.mean[1, :]), axis= 1)
        
        y_pred = np.argmax(np.concatenate([log_prob_0, log_prob_1], axis= 1), axis= 0)
        
        accuracy = np.mean(y_pred == y)
        
        return y_pred, accuracy
        
        
        
        
        
        
        
        
        
        
        
        