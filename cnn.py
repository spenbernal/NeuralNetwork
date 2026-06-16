import numpy as np
class CNN:
    def __init__(self, batch_size, kernel, pool, rows, cols, in_channels, out_channels, kernel_size, padding) -> None:
        # outchannels = (K, 1)
        self.kernel = kernel
        self.pool = pool
        
    def forward(self, X):
        # X has shape (H, W, C)
        # convolve -> pool -> flatten -> feed into NN
        X_1 = convolution2d(X, self.kernel)
        X_2 = maxPooling(X_1, self.pool)
        X_in = X_2.flatten()
        
        return

def convolution1d(X, W):
    # X: (M,)
    # W: (K,)
    Y = np.zeros(len(X) - len(W) + 1)
    k = len(W)
    N = len(Y)
    for i in range(N):
        Y[i] = W @ X[i:i+k]
               
def convolution2d(X, K):
    # X: (H,W)
    # K: filter (K,K)
    h,w = K.shape[0], K.shape[1]
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) 
    k = len(K)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (K * X[i:i+k, j:j+k]).sum()

def maxPooling(X, k):
    H_out = (X.shape[0] - k) + 1
    W_out = (X.shape[1] - k) + 1
    Y = np.zeros((H_out, W_out))
    M, N = Y.shape[0], Y.shape[1]
    for i in range(M):
        for j in range(N):
            Y[i,j] = np.max(X[i:i+k, j:j+k])
            
    return Y
    