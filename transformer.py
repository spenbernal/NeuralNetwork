import numpy as np

class Attention:
    def __init__(self, M, V, q, v) -> None:
        # shape of X: (M,V)
        # hidden dims: q
        self.M = M 
        self.v = V
        self.W_q = np.random.randn(self.v, q)
        self.W_k = np.random.randn(self.v, q)
        self.W_v = np.random.randn(self.v, v)

    def attention(self, X):
        Q = X @ self.W_q # M,q
        K = X @ self.W_k # M,q
        V = X @ self.W_v # M,v
        
        attn = softmax((Q @ K.T) / np.sqrt(K.shape[1])) @ V # M,v
        return attn
            

def softmax(logits):
    # prevent overflow
    logits = logits - logits.max()
    
    probs = np.exp(logits) / np.exp(logits).sum()
    return probs 