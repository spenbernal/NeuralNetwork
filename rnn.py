import numpy as np
import layers as l
from abc import ABC, abstractmethod    
class Decoder(ABC):
    'Abstract class for Decoders'
    hidden_units : int
    D: int
    
    @abstractmethod
    def forward(self, c, y) -> list[np.ndarray]:
        'Forward pass, returns all hidden states'
        pass
    
    @abstractmethod
    def backwards(self, delta):
        'Backpropagation'
        pass
    
    @abstractmethod
    def update(self, gradients, eta) -> None:
        'Optimization to update the parameters'
        pass
    
    @abstractmethod 
    def zero_grad(self):
        'reset all gradients'
        pass
    
class RNNDecoder(Decoder):
    # vec -> seq
    def __init__(self, D, T, C, rnn_cell: l.RNNCell, layer_o: l.Linear) -> None:
        self.rnn_cell = rnn_cell
        self.layer_o = layer_o
        self.D = D
        self.T = T
        self.C = C

    def forward(self, h_enc_T, y):
        # h_enc_T: (N,H_dec) final hidden state
        # y: (N,T)labels (teacher forcing)
        
        # reset every input cache
        self.rnn_cell.reset_cache()
        self.layer_o.reset_cache()
        # init hidden states list
        self.N = h_enc_T.shape[0]
        self.hidden_states = [h_enc_T] # N, H
        
        # one hot encodings for target labels for teacher forcing
        one_hot = np.eye(self.C)
        for t in range(self.T):
            # teacher forcing
            label = np.zeros((self.N, self.C)) if t == 0 else one_hot[y[:,t-1]] #N,C
            self.x_t = np.concatenate([h_enc_T, label], axis= 1)
            # comp. hidden state
            h_t = self.rnn_cell.forward(self.x_t, self.hidden_states[-1])
            self.hidden_states.append(h_t)
            # comp. output correspond to curr hidden state
        H = np.stack(self.hidden_states[1:], axis= 1) 
        self.outputs = self.layer_o.forward(H)
        # outputs: NxTxC
        # hidden_states: Nx(T+1)xh
        return np.array(self.outputs)
            
    def backwards(self, delta):
        # delta: N,T,C , output errors
        # delta of prev hidden state for next iter
        self.delta_future = np.zeros((self.N, self.rnn_cell.h))
        # list storing grad wrt input at each iter
        self.delta_xs = np.zeros((self.N, self.T, self.D))
        self.delta_hs = self.layer_o.backwards(delta)
        for t in range(self.T - 1, -1, -1):
            delta_h_total = self.delta_hs[:, t, :] + self.delta_future
            self.delta_x, self.delta_future = self.rnn_cell.backwards(delta_h_total)
            self.delta_xs[:, t, :] = self.delta_x[:, :self.D]
            
        return self.delta_xs.sum(axis= 1), self.delta_future

    def update(self, eta):
        self.rnn_cell.update(eta)
        self.layer_o.update(eta)
        return 
    
    def zero_grad(self):
        self.rnn_cell.zero_grad()
        self.layer_o.zero_grad()

class AttentionDecoder(Decoder): 
    def __init__(self, D, T, C, rnn_cell: l.RNNCell, layer_o: l.Linear, attention: l.RNNAttention) -> None:
        self.D = D
        self.T = T
        self.C = C
        self.rnn_cell = rnn_cell
        self.layer_o = layer_o
        self.attention = attention
        
    def forward(self, encoder_states, y):
        # encoder_states: N, T, H_enc context vector
        # y: N,T_dec labels (token form)
        
        # reset every input cache
        self.rnn_cell.reset_cache()
        self.layer_o.reset_cache()
        self.attention.set_encoder_states(encoder_states)
        # init hidden states list
        self.N = encoder_states.shape[0]
        self.hidden_states = [encoder_states[:, -1, :]] # N, H
        self.contexts = []
        one_hot = np.eye(self.C)
        for t in range(self.T):
            label = np.zeros(self.C) if t == 0 else one_hot[y[:, t-1]] #N,C
            C_t = self.attention.forward(self.hidden_states[-1]) 
            self.contexts.append(C_t)
            X_t = np.concatenate([C_t, label], axis= 1)
            h_t = self.rnn_cell.forward(X_t, self.hidden_states[-1])
            self.hidden_states.append(h_t)
        
        H = np.stack(self.hidden_states[1:], axis= 1)
        logits = self.layer_o.forward(H)
        return logits
            
    def backwards(self, delta):
        H = self.rnn_cell.h
        self.delta_future = np.zeros((self.N, H))
        
        self.delta_hs = self.layer_o.backwards(delta)
        for t in range(self.T - 1, -1, -1):
            delta_h_total = self.delta_hs[:, t, :] + self.delta_future
            
            delta_inp, delta_h_prev = self.rnn_cell.backwards(delta_h_total)
            
            delta_C_t = delta_inp[:, :H]
            
            d_Q_prev = self.attention.backwards(delta_C_t)
            self.delta_future = d_Q_prev + delta_h_prev
            
        delta_enc_states = self.attention.backwards_final()
        delta_enc_states[:, -1, :] += self.delta_future
        return delta_enc_states
    
    def update(self, eta):
        self.rnn_cell.update(eta)
        self.attention.update(eta)
        self.layer_o.update(eta)
        
    
    def zero_grad(self):
        self.rnn_cell.zero_grad()
        self.layer_o.zero_grad()
        self.attention.zero_grad()
     
class Encoder(ABC):
    'Abstract class for Encoders'
    
    hidden_units : int
    hidden_states: list[np.ndarray]
    
    @abstractmethod
    def forward(self, X) -> list[np.ndarray]:
        'Forward pass, returns all hidden states'
        pass
    
    @abstractmethod
    def backwards(self, delta_enc):
        'Backpropagation'
        pass
    
    @abstractmethod
    def update(self, gradients, eta) -> None:
        'Optimization to update the parameters'
        pass
class RNNEncoder(Encoder):
    # seq -> vec
    def __init__(self, T, D, rnn_cell: l.RNNCell) -> None:
        # T: len of seq
        # D: dim of seq
        self.T = T
        self.D = D
        self.rnn_cell = rnn_cell
       
        
    def forward(self, X):
        # X: (N,T,D)
        self.rnn_cell.reset_cache()
        
        self.X = X
        self.hidden_states = [np.zeros((self.X.shape[0], self.rnn_cell.h))]
        for t in range(self.T):
            x_t = X[:,t,:] #N,D
            h_t = self.rnn_cell.forward(x_t, self.hidden_states[-1])
            self.hidden_states.append(h_t)
            
        h_T = self.hidden_states[-1]
        self.encoder_states = np.stack(self.hidden_states, axis= 1)
        return h_T
    
    def backwards(self, delta_enc):
        N = delta_enc.shape[0]
        self.delta_future = np.zeros((N, self.rnn_cell.h))  
        self.delta_xs = np.zeros_like(self.X)
        for t in range(self.T - 1, -1, -1):
            delta_h_t = delta_enc[:, t, :] + self.delta_future
            self.delta_x, self.delta_future = self.rnn_cell.backwards(delta_h_t)
            self.delta_xs[:, t, :] = self.delta_x
            
        return self.delta_xs, self.delta_future
    
    def update(self, eta):
        self.rnn_cell.update(eta)
        
    def zero_grad(self):
        self.rnn_cell.zero_grad()
class seq2vec:
    def __init__(self, encoder: RNNEncoder, layer_o: l.Linear) -> None:
        self.encoder = encoder
        self.layer_o = layer_o
        
    def forward(self, X):
        # X: (N, T, D)
        self.h_T = self.encoder.forward(X) # N,h
        output = self.layer_o.forward(self.h_T) # N, o
        return output
    
    def backwards(self, delta):
        # delta: N,O
        delta_o = self.layer_o.backwards(delta)
        delta_xs, delta_h = self.encoder.backwards(delta_o)   
        return delta_xs, delta_h
    
    def update(self, eta):
        self.encoder.update(eta)
        self.layer_o.update(eta)
        return
        
    def zero_grad(self):
        self.encoder.zero_grad()
        self.layer_o.zero_grad()
        return 
      
class seq2vecBidirectional:
    def __init__(self, f_rnn: RNNEncoder, b_rnn: RNNEncoder, out_dim) -> None:
        self.f_rnn = f_rnn
        self.b_rnn = b_rnn
        self.h = self.f_rnn.rnn_cell.h
        self.layer_o = l.Linear(2*self.h, out_dim)    
        
    def forward(self, X):
        f_h_T = self.f_rnn.forward(X) #N,H
        b_h_T = self.b_rnn.forward(X[:, ::-1, :]) #N,H
                
        self.h_T = np.concatenate([f_h_T, b_h_T], axis= 1)
        logits = self.layer_o.forward(self.h_T)
        return logits
        
    
    def backwards(self, delta):
        delta_o = self.layer_o.backwards(delta) #N, 2H
        delta_f = delta_o[:, :self.h]
        delta_b = delta_o[:, self.h:]
        delta_f_in, delta_f_next = self.f_rnn.backwards(delta_f) 
        delta_b_in_rev, delta_b_next = self.b_rnn.backwards(delta_b)
        delta_b_in = delta_b_in_rev[:, ::-1, :]
        return delta_f_in, delta_f_next, delta_b_in, delta_b_next
    
    def update(self, eta):
        self.f_rnn.update(eta)
        self.b_rnn.update(eta)
        self.layer_o.update(eta)
        
    def zero_grad(self):
        self.f_rnn.zero_grad()
        self.b_rnn.zero_grad()
        self.layer_o.zero_grad()

class seq2seq:
    # encoder -> context vector -> decoder
    def __init__(self, encoder: RNNEncoder, decoder: RNNDecoder) -> None:
        self.encoder = encoder
        self.decoder = decoder
        # linear projection layer
        self.proj = l.Linear(self.encoder.rnn_cell.h, self.decoder.D, bias= False)

    def forward(self, X, y):
        # X: N, T, D sequence of length T with D dim
        # y: N, T', target labels in token form
        self.h_T = self.encoder.forward(X)
        c = self.proj.forward(self.h_T)
        outputs = self.decoder.forward(c, y)
        return outputs
    
    def backwards(self, delta):
        grad_wrt_inp, delta_d = self.decoder.backwards(delta)
        delta = self.proj.backwards(delta_d)
        grad_wrt_inp, delta_e = self.encoder.backwards(delta)
        
        return grad_wrt_inp, delta_e
        
    def update(self, eta):
        self.encoder.update(eta)
        self.decoder.update(eta)
        self.proj.update(eta)
        
    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad() 
        self.proj.zero_grad()    
        
class Attentionseq2seq:
    def __init__(self, encoder: RNNEncoder, decoder: AttentionDecoder) -> None:
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, X, y):
        # X: (N, T, D)
        # y: (N, T') labels
        self.encoder.forward(X)
        encoder_states =  self.encoder.encoder_states # N, T_enc, H_enc
        outputs = self.decoder.forward(encoder_states, y) # N, T_dec, C
        
        return outputs
    
    def backwards(self, delta):
        delta_enc = self.decoder.backwards(delta)
        delta_inp, delta_h = self.encoder.backwards(delta_enc)
        return delta_inp, delta_h
    
    def update(self, eta):
        self.encoder.update(eta)
        self.decoder.update(eta)
        
    def zero_grad(self):
        self.encoder.zero_grad()
        self.decoder.zero_grad()