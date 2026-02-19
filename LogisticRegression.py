import numpy as np

class LogisticRegression:
    def __init__(self, features) -> None:
        self.features = features
        self.weights = np.random.randn(self.features)
        self.bias = np.random.randn()
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def logloss(self, y_true, y_pred):
        return  -1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def forward(self, X_train):
        z = X_train @ self.weights + self.bias 
        z = self.sigmoid(z)
        return z
    
    def gd(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b
        return 
    
    def train(self, X_train, y_train, epochs, lr):
        N = X_train.shape[0]
        for e in range(epochs):
            indices = np.random.permutation(N)
            epoch_loss = 0.0
            for idx in indices:
                x = X_train[idx]
                y = y_train[idx]
                
                y_preds = self.forward(x)
                loss = self.logloss(y, y_preds)
                epoch_loss += loss
                grad_w = (y_preds - y) * x
                grad_b = (y_preds - y)
                #grad_w = x.T @ (y_preds - y_train[idx]) / N # (N,1) . (N, dim)
                #grad_b = np.sum((y_preds - y_train[idx])) / N
                self.gd(grad_w, grad_b, lr)
            
            print(f'Epoch {e+1} | Loss {(epoch_loss / N):.4f}')
        print(f'{epochs} epochs finished!')
        return
        
    def eval(self, X_test, y_test):
        N = y_test.shape[0]
        y_pred = self.forward(X_test)
        loss = np.mean([self.logloss(y_test[i], y_pred[i]) for i in range(N)])

        print(f'Loss: {loss:.4F}')
        
        return    
    
        