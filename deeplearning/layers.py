import numpy as np
from abc import ABC, abstractmethod   
class Layer(ABC):
    
    @abstractmethod
    def forward(self, X):
        pass
    
    @abstractmethod
    def backwards(self, delta):
        pass
    
    @abstractmethod
    def update(self, eta):
        pass
    
    @abstractmethod
    def zero_grad(self):
        pass


class Linear(Layer):
    def __init__(self, D, O, bias= True) -> None:
        # D: dim of input
        # O: dim of output
        self.D = D
        self.O = O
        self.W = np.random.randn(O, D) * np.sqrt(2.0 / D)
        self.b = np.zeros(O)
        self.inputs = []
        self.use_bias = bias
    
    def forward(self, X):
        # X: (N,D) input vector
        self.inputs.append(X) 
        a = X @ self.W.T + self.b 
        return a
    
    def backwards(self, delta): 
        # delta: (N,O) error from future layer
        X = self.inputs.pop()
        X_flat = X.reshape(-1, self.D)
        delta_flat = delta.reshape(-1, self.O)
        
        self.grad_W += delta_flat.T @ X_flat
        self.grad_b += delta_flat.sum(axis= 0)
        
        # update delta
        delta = delta @ self.W
        return delta
    
    def update(self, eta):
        self.W -= eta * self.grad_W
        if self.use_bias:
            self.b -= eta * self.grad_b
        return 
    
    def zero_grad(self):
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        return 
    
    def reset_cache(self):
        self.inputs = []
        return
    
class ReLU(Layer):
    def forward(self, logits):
        self.logits = logits
        return np.maximum(0, logits)
    
    def backwards(self, delta):
        
        delta = delta * np.where(self.logits > 0, 1, 0)
        return delta
    
    def update(self, eta):
        pass
    
    def zero_grad(self):
        pass
    
class Sigmoid(Layer):
    def forward(self, z):
        self.z = z
        return (1 / 1 + np.exp(-z))
    
    def backwards(self, delta):
        return delta * ((1 / 1 + np.exp(-self.z))) * (1-(1 / 1 + np.exp(-self.z)))
    
    def update(self, eta):
        pass
    
    def zero_grad(self):
        pass
    
class Dropout(Layer):
    def __init__(self, p) -> None:
        self.p = p
        
    def forward(self, logits: np.ndarray, training):
        if not training:
            return logits
    
        self.mask = np.random.random(size= logits.shape) > self.p
        return logits * self.mask / (1 - self.p)
    
    def backwards(self, delta):
        delta = delta * self.mask / (1 - self.p)
        return delta 
    
    def update(self, eta):
        pass
    
    def zero_grad(self):
        pass
    
class ConvolutionLayer(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding) -> None:
        # out_channels: number of learnable filters
        # k: dim of filter
        self.k = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = np.random.randn(out_channels, in_channels, self.k, self.k) \
                * np.sqrt(2.0 / in_channels * kernel_size * kernel_size)
        self.b = np.zeros(out_channels)
        self.p = padding
    
    def forward(self, X):
        # X: (N,in_ch, H, W)
        self.X = X
        self.X_pad = np.pad(X, ((0,0), (0,0), (self.p, self.p), (self.p, self.p)), mode='constant')
        Y = np.zeros((X.shape[0], self.out_channels, X.shape[2] + (2 * self.p) - self.k + 1, X.shape[3] + (2 * self.p) - self.k + 1)) 
        for c in range(self.out_channels): 
            kernel = self.W[c, :, :, :] #in, k, k
            for i in range(Y.shape[2]):
                for j in range(Y.shape[3]):
                    x_patch = self.X_pad[:, :, i:i+self.k, j:j+self.k] # N, C, k, k
                    Y[:, c, i, j] = (kernel * x_patch).sum(axis= (1,2,3)) + self.b[c]
        
        return Y
        
    def backwards(self, delta): 
        # delta: (N, out, Y.shape[1], Y.shape[2])
        
        for c in range(self.out_channels):
            for i in range(delta.shape[2]):
                for j in range(delta.shape[3]):
                    self.grad_w[c] += (delta[:, c, i, j][:, None, None, None] * self.X_pad[:, :, i:i+self.k, j:j+self.k]).sum(axis= 0)
                    self.grad_b[c] += delta[:, c, i, j].sum(axis= 0)
                    self.grad_x_pad[:, :, i:i+self.k, j:j+self.k] += delta[:, c, i, j][:, None, None, None] * self.W[c, :, :, :] #N,1,1,1 * 1,C_in,k,k
        
        grad_x = self.grad_x_pad[:, :, self.p:-self.p, self.p:-self.p] if self.p > 0 else self.grad_x_pad

        return grad_x
    
    def update(self, eta):
        self.W -= eta * self.grad_w
        self.b -= eta * self.grad_b
        return
    
    def zero_grad(self):
        self.grad_w = np.zeros_like(self.W) #C_out, C_in, k, k
        self.grad_b = np.zeros_like(self.b)
        self.grad_x_pad = np.zeros_like(self.X_pad)
        return 
    
class MaxPooling(Layer):
    def __init__(self, k) -> None:
        self.k = k
        
    def forward(self, X):
        # X: N, in_channels, H, W
        self.X = X
        H_out = (X.shape[2] - self.k) + 1
        W_out = (X.shape[3] - self.k) + 1
        self.Y = np.zeros((X.shape[0], X.shape[1], H_out, W_out))
        
        for i in range(H_out):
            for j in range(W_out):
                X_patch = X[:,:, i:i+self.k, j:j+self.k] #N,C_in, k, k
                self.Y[:,:,i,j] = np.max(X_patch, axis= (2,3))
        
        return self.Y
    
    def backwards(self, delta):
        # delta: (N, C, Y.shape[1], Y.shape[2])
        grad_x = np.zeros_like(self.X) #N,C_in,H,W
        for i in range(delta.shape[2]):
            for j in range(delta.shape[3]):
                X_patch = self.X[:, :, i:i+self.k, j:j+self.k] #N, C_in, k,k
                max_vals = self.Y[:, :, i, j][:, :, None, None] # (N,C_in,1,1)
                mask = (X_patch == max_vals) # C,k,k
                
                grad_x[:, :, i:i+self.k, j:j+self.k] += delta[:, :, i, j][:, :, None, None] * mask
            
        return grad_x

    def update(self, eta):
        pass
    
    def zero_grad(self):
        pass
    
class Flatten(Layer):
    def forward(self, X):
        self.input_shape = X.shape
        # N,D
        return X.reshape(X.shape[0], -1)
    
    def backwards(self, delta):
        return delta.reshape(self.input_shape)
    
    def update(self, eta):
        pass
    
    def zero_grad(self):
        pass

class Upsample:
    def __init__(self, scale) -> None:
        self.scale = 2
        
    def forward(self, X):
        self.input_shape = X.shape
        
        X = np.repeat(X, self.scale, axis= 2)
        X = np.repeat(X, self.scale, axis= 3)
        
        return X
    
    def backwards(self, delta):
        N, C, H, W = self.input_shape
        delta = delta.reshape(N, C, H, self.scale, W, self.scale)
        return delta.sum(axis= (3,5))  
    
    def update(self, eta):
        pass
    
    def zero_grad(self):
        pass      

class RNNCell(Layer):
    def __init__(self, D, h) -> None:
        self.h = h
        self.W_xh = np.random.randn(h, D)
        self.W_hh = np.random.randn(h,h)
        self.b_h = np.zeros(h)
        self.cache = []
        
    def forward(self, X_t, h_prev):
        # x_t: (N,D) 
        # h_prev: (N,h)
        a_t = X_t @ self.W_xh.T  + h_prev @ self.W_hh.T + self.b_h  #N,h
        self.cache.append((X_t, h_prev, a_t))
        self.h_t = np.maximum(0, a_t)
        return self.h_t
    
    def backwards(self, delta):
        # delta (dL/dh_t): N,h
        X_t, h_prev, a_t = self.cache.pop() #N,D , N,h , N,h
        delta = delta * np.where(a_t > 0, 1, 0)
        self.grad_W_xh += delta.T @ X_t
        self.grad_W_hh += delta.T @ h_prev 
        self.grad_b_h += delta.sum(axis= 0)
        
        delta_h = delta @ self.W_hh
        delta_x = delta @ self.W_xh
        return delta_x, delta_h
    
    def update(self, eta):
        self.W_hh -= eta * self.grad_W_hh
        self.W_xh -= eta * self.grad_W_xh
        self.b_h -= eta * self.grad_b_h
        return 
    
    def zero_grad(self):
        self.grad_W_xh = np.zeros_like(self.W_xh)
        self.grad_W_hh = np.zeros_like(self.W_hh)
        self.grad_b_h = np.zeros_like(self.b_h)
        
    def reset_cache(self):
        self.cache = []

class RNNAttention:
    def __init__(self, H_enc_d, H_dec_d) -> None:
        # initialize projection matrices for Key, Values
        self.proj_K = Linear(H_enc_d, H_dec_d, bias= False)
        self.proj_V = Linear(H_enc_d, H_dec_d, bias= False)
        
    def set_encoder_states(self, H_enc):
        # project encoder hidden states to same space as query space
        # intialize lists for queries and attention weights
        # initalize total gradient of keys and values
        self.K = self.proj_K.forward(H_enc) 
        self.V = self.proj_V.forward(H_enc)
        self.scale = np.sqrt(self.K.shape[-1])
        self.Qs = []
        self.alphas = []
        self.d_K_total = np.zeros_like(self.K)
        self.d_V_total = np.zeros_like(self.V)
        
    def forward(self, H_dec: np.ndarray):
        # H_dec: N, H_dec
        self.Qs.append(H_dec)
        # calculate dot product of query and keys
        self.dot_prod = (H_dec[:, None, :] @ self.K.transpose(0, 2, 1)).squeeze(axis= 1) # N,1,H_dec x N,H_dec, T_enc
        # scale dot product (scaled dot product attn)
        scores = self.dot_prod / self.scale # N,T_enc
        # compute attention weights (softmax)
        alphas = np.exp(scores) / np.exp(scores).sum(axis= 1, keepdims= True) # N,T_enc
        # store weights
        self.alphas.append(alphas)
        # calculate attention score
        attn = (alphas[:, None, :] @ self.V).squeeze(axis= 1) # N,1,T_enc x N,T_enc,H_dec
        # return attention output/context vector
        return attn
    
    def backwards(self, delta):
        # delta: N,H_dec
        # retrieve current attention weight and query
        alphas = self.alphas.pop()
        Q = self.Qs.pop()
        
        delta_attn = delta[:, None, :] # N, 1, H_dec
        # compute gradient wrt Values
        d_V = alphas[:, None, :].transpose(0, 2, 1) @ delta_attn #N,T_enc,H_dec
        # accumulate gradient
        self.d_V_total += d_V
        # compute gradient wrt attention weights
        d_alphas = (self.V @ delta_attn.transpose(0, 2, 1)).squeeze(axis= 2) #N,T_enc
        # compute gradient wrt scaled dot product (softmax VJP)
        d_scores = alphas * (d_alphas  - np.sum(d_alphas * alphas, axis= 1, keepdims= True)) #N,T_enc
        # compute gradient wrt unscaled scores
        d_dot = d_scores / self.scale #N,T_enc
        # compute gradient wrt Keys
        d_K_scores = d_dot[:,:, None] @ Q[:, None, :] # N,T_enc,H_dec
        self.d_K_total += d_K_scores
        # compute gradient wrt Query
        d_Q = (d_dot[:,:, None].transpose(0, 2, 1) @ self.K).squeeze(axis= 1) #N,H_dec
        return d_Q
        
    def backwards_final(self):
        # pass accumulated gradients back to projection layers
        d_H_enc_K = self.proj_K.backwards(self.d_K_total)
        d_H_enc_V = self.proj_V.backwards(self.d_V_total)
        return d_H_enc_K + d_H_enc_V
        
    def zero_grad(self):
        self.proj_K.zero_grad()
        self.proj_V.zero_grad()
        
    def update(self, eta):
        self.proj_K.update(eta)
        self.proj_V.update(eta)
    
class TransformerAttention:
    def __init__(self, D_q, D_k, dim_k, dim_v) -> None:
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.proj_Q = Linear(D_q, self.dim_k, bias= False)
        self.proj_K = Linear(D_k, self.dim_k, bias= False)
        self.proj_V = Linear(D_k, self.dim_v, bias= False)
        self.alphas = []
        
        
    
    def forward(self, X_q, X_k, X_v, causal_mask= False):
        
        self.Q = self.proj_Q.forward(X_q) # N, T, dim_k
        self.K = self.proj_K.forward(X_k)
        self.V = self.proj_V.forward(X_v) # N, T, dim_v
        self.scale = np.sqrt(self.K.shape[-1])
        dot = self.Q @ self.K.transpose(0, 2, 1) # N, T, T
        score = dot / self.scale # N, T_q, T_k
        if causal_mask:
            T_q, T_k = score.shape[1], score.shape[2]
            mask = np.tril(np.ones((T_q, T_k))).astype(bool)
            score[:, ~mask] = -1e7
        alpha = np.exp(score) / np.exp(score).sum(axis= -1, keepdims= True) # N, T, T
        self.alphas.append(alpha)
        attn = alpha @ self.V # N, T, dim_v
        return attn
    
    def backwards(self, delta):
        # delta: N, T, dim_v
        alpha = self.alphas.pop()
        d_V = alpha.transpose(0, 2, 1) @ delta
        d_alpha = delta @ self.V.transpose(0, 2, 1) # N, T, T
        d_score = alpha * (d_alpha  - np.sum(d_alpha * alpha, axis= -1, keepdims= True))
        d_dot = d_score / self.scale # N, T, T
        d_K = d_dot.transpose(0, 2, 1) @ self.Q # N, T, T x N, T, dim_k
        d_Q = d_dot @ self.K # N, T, T x N, T, dim_k
        
        d_X_q = self.proj_Q.backwards(d_Q) 
        d_X_k = self.proj_K.backwards(d_K)  
        d_X_v = self.proj_V.backwards(d_V)
        
        return d_X_q, d_X_k, d_X_v
    
    def zero_grad(self):
        self.proj_Q.zero_grad()
        self.proj_K.zero_grad()
        self.proj_V.zero_grad()
    
    
    def update(self, eta):
        self.proj_Q.update(eta)
        self.proj_K.update(eta)
        self.proj_V.update(eta)
    
    def reset_cache(self):
        self.alphas = []
        
        
class MultiHeadAttention:
    def __init__(self, H, D_q, D_kv, d_k, d_v, p_o) -> None:
        # H: num of attention heads
        # D_q: query dim
        # D_kv: key & val dim
        # d_k: key and query len
        # d_v: val len
        # p_o: projection output after concatenation
        self.mha = [TransformerAttention(D_q, D_kv, d_k, d_v) for _ in range(H)]
        self.proj_o = Linear(H * d_v, p_o, bias= False)
        self.H = H
        self.d_k = d_k
        self.d_v = d_v
        self.p_o = p_o
        
    
    def forward(self, X_q, X_k, X_v, causal_mask= False):
        # X_q: N, T_q, D_q
        # X_k: N, T_k, D_k
        # X_v: N, T_k, D_k
        
        self.X_q = X_q
        self.X_k = X_k
        self.X_v = X_v
        self.outputs = [] 
        for head in self.mha:
            # self attention in encoder
            out = head.forward(X_q, X_k, X_v, causal_mask) # N, T, d_v
            self.outputs.append(out)
        self.outs = np.concatenate(self.outputs, axis= -1) # N, T, H*d_v
        
        return self.proj_o.forward(self.outs) 
    
    def backwards(self, delta):
        delta_o = self.proj_o.backwards(delta)
        delta_H = np.split(delta_o, self.H, axis= -1)
        
        delta_X_q_total = np.zeros_like(self.X_q)
        delta_X_k_total = np.zeros_like(self.X_k)
        delta_X_v_total = np.zeros_like(self.X_v)
        for idx, head in enumerate(self.mha):
            delta_head = delta_H[idx]
            d_X_q, d_X_k, d_X_v = head.backwards(delta_head)
            delta_X_q_total += d_X_q
            delta_X_k_total += d_X_k
            delta_X_v_total += d_X_v
            
        return delta_X_q_total, delta_X_k_total, delta_X_v_total
    
    def update(self, eta):
        self.proj_o.update(eta)
        for head in self.mha:
            head.update(eta)
        
    def zero_grad(self):
        self.proj_o.zero_grad()
        for head in self.mha:
            head.zero_grad()
    
    def reset_caches(self):
        for head in self.mha:
            head.reset_cache()
        
class Embedding:
    def __init__(self, vocab_size, D) -> None:
        # map onehot -> dense vector D
        self.W = np.random.randn(vocab_size, D)
        
        
    def forward(self, tokens: np.ndarray):
        # tokens: N,T
        self.tokens = tokens
        return self.W[tokens] # N, T, D
    
    def backwards(self, delta):
        # delta: N,T,D
        self.grad_W.fill(0)
        np.add.at(self.grad_W, self.tokens, delta)
        # no grad wrt token ids
        return 
    
    def update(self, eta):
        self.W -= eta * self.grad_W
        
    def zero_grad(self):
        self.grad_W = np.zeros_like(self.W)
        
class PositionalEncoding:
    def __init__(self, max_len, D) -> None:
        # max len: maximum length in sequence
        # D: embedding dim
        # position in seq 
        pos = np.arange(max_len)[:, None]
        j = np.arange(D)[None, :]
        
        angles = pos / np.power(10000, 2 * (j//2) / D) # max_len, D
        self.P = np.zeros(max_len, D)
        self.P[:, 0::2] = np.sin(angles[:, 0::2])
        self.P[:, 1::2] = np.cos(angles[:, 1::2])
    
    def forward(self, X):
        return X + self.P[None, :, :]
    
    def backwards(self, delta):
        return delta
             
class LayerNorm:
    def __init__(self, normalization_axes, eps= 1e-5) -> None:
        self.normalization_axes = normalization_axes
        self.eps = eps
        
    def forward(self, X):
        
        self.X = X
        self.mu = X.mean(axis= self.normalization_axes, keepdims= True)
        self.var = X.var(axis= self.normalization_axes, keepdims= True)
        self.X_norm = (self.X - self.mu) / np.sqrt(self.var + self.eps)
        return self.X_norm
        
    def backwards(self, delta):
        # delta: same shape as X
        axes = self.normalization_axes
        if isinstance(axes, int):
            axes = (axes,)
        
        m = np.prod([self.X.shape[axis] for axis in axes])
        dX = (1 / m) * (1 / np.sqrt(self.var + self.eps)) * (
            m * delta - np.sum(delta, axis= axes, keepdims= True) 
            - self.X_norm * np.sum(delta * self.X_norm, axis=axes, keepdims=True) 
        )
        return dX