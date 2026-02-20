import numpy as np

class LinearRegressor:
    # Constructor
    def __init__(self, n_features):
        self.weights = np.random.rand(n_features)
        self.bias = np.random.rand()
        
    # Computation
    def forward(self, X_train):
        return X_train @ self.weights + self.bias 
    
    def gd(self, grad_w, grad_b, lr):
        self.weights -= lr * grad_w
        self.bias -= lr * grad_b
        return 
    
    # Training Function
    def train(self, X_train, y_train, epochs=1000, lr=.001, verbose=False):
        N = X_train.shape[0]
        for e in range(epochs):
            y_pred = self.forward(X_train)
            loss = self.MSE(y_train, y_pred)
            grad_w = (2/N) * (X_train.T @ (y_pred - y_train)) 
            grad_b = (2/N) * np.sum((y_pred - y_train))
            self.gd(grad_w, grad_b, lr)     
            
            print(f'********* Epoch {e+1} *********') 
            print(f'Loss: {loss:.4f}')    
        print(f'{epochs} epochs done!') 
    # Loss
    def MSE(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    # Evaluation
    def evaluate(self, X_test, y_test):
        y_pred = self.forward(X_test)
        loss = self.MSE(y_test, y_pred)
        print(f'Evaluation')
        print(f'MSE: {loss:.4f}')
        print('Parameters')
        for idx, w in enumerate(self.weights):
            print(f'Weight {idx+1}: {w:.3f}')
        print(f'Bias: {self.bias:.3f}')
    