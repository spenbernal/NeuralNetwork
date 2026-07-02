import numpy as np
import nn as nn
class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, k) -> None:
        # out_channels: number of learnable filters
        # k: dim of filter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = np.random.randn(out_channels, in_channels, k, k) 
        self.b = np.zeros(out_channels)
        self.k = k
    
    def forward(self, X):
        # X: (in_ch, H, W)
        self.X = X
        Y = np.zeros((self.out_channels, X.shape[1] - self.k + 1, X.shape[2] - self.k + 1)) 
        for c in range(Y.shape[0]): 
            kernel = self.W[c, :, :, :] #in, k, k
            for i in range(Y.shape[1]):
                for j in range(Y.shape[2]):
                    x_patch = X[:, i:i+self.k, j:j+self.k]
                    Y[c, i, j] = (kernel * x_patch).sum() + self.b[c]
        
        return Y
        
    def backwards(self, delta): 
        # delta: (out, Y.shape[1], Y.shape[2])
        self.grad_w = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        self.grad_x = np.zeros_like(self.X)
        
        for c in range(self.out_channels):
            for i in range(delta.shape[1]):
                for j in range(delta.shape[2]):
                    self.grad_w[c] += delta[c, i, j] * self.X[:, i:i+self.k, j:j+self.k]
                    self.grad_b[c] += delta[c, i, j]
                    self.grad_x[:, i:i+self.k, j:j+self.k] += delta[c, i, j] * self.W[c, :, :, :]

        return self.grad_x
    
    def update(self, eta):
        self.W -= eta * self.grad_w
        self.b -= eta * self.grad_b
        return

class MaxPooling:
    def __init__(self, k) -> None:
        self.k = k
        
    def forward(self, X):
        # X: in_channels, H, W
        self.X = X
        H_out = (X.shape[1] - self.k) + 1
        W_out = (X.shape[2] - self.k) + 1
        self.Y = np.zeros((X.shape[0], H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                X_patch = X[:, i:i+self.k, j:j+self.k] # in, k, k
                self.Y[:,i,j] = np.max(X_patch, axis= (1,2))
        
        return self.Y
    
    def backwards(self, delta):
        # delta: (C, Y.shape[1], Y.shape[2])
        grad_x = np.zeros_like(self.X)
        for i in range(delta.shape[1]):
            for j in range(delta.shape[2]):
                X_patch = self.X[:, i:i+self.k, j:j+self.k] #C, k,k
                max_vals = self.Y[:, i, j][:, None, None] # (C,1,1)
                mask = (X_patch == max_vals) # C,k,k
                
                grad_x[:, i:i+self.k, j:j+self.k] += delta[:, i, j][:, None, None] * mask
            
        return grad_x

    def update(self, eta):
        pass
class Flatten:
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(-1)
    
    def backwards(self, delta):
        return delta.reshape(self.input_shape)
    
    def update(self, eta):
        pass
