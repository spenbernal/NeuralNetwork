import numpy as np
class vec2seq:
    # decoder
    def __init__(self, D, T, C, h) -> None:
        # D: X dim
        # h: hidden units
        # T: output sequence length (fixed?)
        # C: dimensions of each step in sequence (vocab size)
        self.T = T
        self.C = C
        self.h = h
        self.w_xh = np.random.randn(h, D + C)
        self.w_hh = np.random.randn(h, h)
        self.w_oh = np.random.randn(C, h)
        self.b_h = np.random.randn(h)
        self.b_o = np.random.randn(C)
    
    def forward(self, X, y):
        # Generate 1 token at a time
        # X: (D, ) input vector
        # y: (T, ) token labels
        # teacher forcing
        self.probs = []
        self.hidden_states = [np.zeros(self.h)]
        self.logits = []
        self.inputs = []
        one_hot = np.eye(self.C)
        for t in range(self.T):
            label = one_hot[y[t-1]] if t != 0 else np.zeros(self.C)
            input = np.concatenate([X, label]) 
            self.inputs.append(input)
            h_prev = self.hidden_states[-1]
            h_t = ReLU(self.w_xh @ input + self.w_hh @ h_prev + self.b_h)
            self.hidden_states.append(h_t)
            
            # get distribution over at step t over tokens
            logits = self.w_oh @ h_t + self.b_o
            probs = softmax(logits) 
            self.probs.append(probs)
            self.logits.append(logits)

        return 
        
    def predict(self, X):
        # Generate 1 token at a time
        # X: (D,)
        probabilities = []
        hidden_states = [np.zeros(self.h)]
        output_tokens = []
        y_prev = np.zeros(self.C)
        for _ in range(self.T):
            input = np.concatenate([X, y_prev]) 
            h_prev = hidden_states[-1]
            h_t = ReLU(self.w_xh @ input + self.w_hh @ h_prev + self.b_h)
            hidden_states.append(h_t)
            # get distribution over at step t over tokens
            probs = softmax(self.w_oh @ h_t + self.b_o) 
            probabilities.append(probs)
            # sample token
            y_t = np.random.choice(len(probs), p= probs)
            # update y_prev for next iteration
            y_prev[:] = 0
            y_prev[y_t] = 1
            output_tokens.append(y_t)
        return output_tokens
    
    def backwards(self, y):
        # X: (D, ) the input vector
        # y: (T, ) target vector (in tokens)
        H = np.array(self.hidden_states[1:]) #TxH
        self.P = np.array(self.probs) # TxC
        self.y = np.eye(self.C)[y] # TxC
        delta_o = (self.P - self.y) / self.T # TxC
        grad_w_oh = delta_o.T @ H # CxH
        grad_b_o = delta_o.sum(axis= 0) # C,
        
        grad_w_xh = np.zeros_like(self.w_xh)
        grad_w_hh = np.zeros_like(self.w_hh)
        grad_b_h = np.zeros_like(self.b_h)
        
        delta_next = np.zeros(H.shape[1])
        for t in range(self.T - 1, -1, -1):
            h_t = self.hidden_states[t+1]
            h_prev = self.hidden_states[t] # H,
            
            delta_h = (self.w_oh.T @ delta_o[t] + self.w_hh.T @ delta_next) * np.where(h_t > 0, 1, 0)
            
            grad_w_hh += np.outer(delta_h, h_prev) # H,H
            grad_w_xh += np.outer(delta_h, self.inputs[t]) # H, x D+C,
            grad_b_h += delta_h
            
            delta_next = delta_h       
        
        gradients = grad_w_xh, grad_w_hh, grad_b_h, grad_w_oh, grad_b_o
        return gradients, delta_next

    def update(self, gradients, eta):
        # gradient descent
        grad_input, grad_hidden, grad_hidden_bias, grad_output, grad_output_bias = gradients
        
        self.w_xh -= eta * grad_input
        self.w_hh -= eta * grad_hidden
        self.w_oh -= eta * grad_output
        self.b_h -= eta * grad_hidden_bias
        self.b_o -= eta * grad_output_bias
    
    def train(self, X, y, epochs, eta):
        for e in range(epochs):
            self.forward(X, y)
            gradients, _ = self.backwards(y)
            loss = ce(self.y, self.P)
            self.update(gradients, eta)
            print(f'Epoch {e+1} | Cross Entropy Loss: {loss}')
            
    def eval(self, X, vocabulary):
        tokens = self.predict(X)
        print('Decoded the following sequence')
        print(*[vocabulary[token] for token in tokens])
class Encoder:
    def __init__(self, M, N, hidden_units) -> None:
        self.M = M
        self.N = N
        self.hidden_units = hidden_units
        self.input_weight = np.random.randn(self.hidden_units, self.N)
        self.hidden_weight = np.random.randn(self.hidden_units, self.hidden_units)
        self.bias_hidden = np.random.randn(self.hidden_units)
    
    def forward(self, X):
        self.hidden_states = [np.zeros(self.hidden_units)]
        for i in range(self.M):
            x = X[i] 
            hidden_state = ReLU(self.input_weight @ x 
                                + self.hidden_weight @ np.array(self.hidden_states[-1]) 
                                + self.bias_hidden)
            
            self.hidden_states.append(hidden_state)
            
        return self.hidden_states[-1]
    
    def backwards(self, delta_enc, X):
        delta = (delta_enc) * np.where(self.hidden_states[-1] > 0, 1, 0) #(hidden,)
        grad_input = np.zeros_like(self.input_weight)
        grad_hidden = np.zeros_like(self.hidden_weight)
        grad_hidden_bias = np.zeros_like(self.bias_hidden)
        
        for t in range(self.M, 0, -1):
            grad_input += np.outer(delta, X[t-1]) 
            grad_hidden += np.outer(delta, self.hidden_states[t-1])
            grad_hidden_bias += delta
            
            delta = (self.hidden_weight.T @ delta) * np.where(self.hidden_states[t-1] > 0, 1, 0)        
            
        return grad_input, grad_hidden, grad_hidden_bias
    
    def update(self, gradients, eta):
        grad_input, grad_hidden, grad_hidden_bias = gradients
        
        self.input_weight -= eta * grad_input
        self.hidden_weight -= eta * grad_hidden
        self.bias_hidden -= eta * grad_hidden_bias
class seq2vec(Encoder):
    # Time series forecast of fish price data
    def __init__(self, M, N, hidden_units, output_dim) -> None:
        super().__init__(M, N, hidden_units)
        self.output_dim = output_dim
        self.output_weight = np.random.randn(self.output_dim, self.hidden_units)
        self.bias_output = np.random.randn(self.output_dim)
        
    def forward(self, X):
        h_T = super().forward(X)
        y_hat = self.output_weight @ h_T + self.bias_output
        return y_hat
    
    def backwards(self, y_hat, y, X):
        delta_o = (y_hat - y) 
        grad_output = np.outer(delta_o, self.hidden_states[-1]) #(output,)x(hidden_dim,) = (output, hidden)
        grad_output_bias = delta_o 
        delta_enc = self.output_weight.T @ delta_o
        
        grad_input, grad_hidden, grad_hidden_bias = super().backwards(delta_enc, X)
        
        gradients = grad_input, grad_hidden, grad_hidden_bias, grad_output, grad_output_bias       
        return gradients
    
    def update(self, gradients, eta):
        # gradient descent
        *enc_grads, grad_output, grad_output_bias = gradients
        
        super().update(enc_grads, eta)
        self.output_weight -= eta * grad_output
        self.bias_output -= eta * grad_output_bias
    
    def train(self, X, y, epochs, eta):
        for e in range(epochs):
            y_hat = self.forward(X)
            loss = MSE(y, y_hat)
            gradients = self.backwards(y_hat, y, X)
            self.update(gradients, eta)
            print(f'Epoch {e} | Training Loss: {loss.round(5)}')

    def predict(self, X):
        prediction = self.forward(X)
        print(f'The prediction for the next price is ${prediction.round(3)}')
        
def ReLU(x):
    return np.maximum(0,x)

def MSE(y, y_hat):
    return 1/2 * (y_hat - y).sum()**2        
class seq2vecBidirectional:
    # bidirectional RNN for sentiment classification
    def __init__(self, f_rnn: Encoder, b_rnn: Encoder, output_dim) -> None:
        self.f_rnn = f_rnn
        self.b_rnn = b_rnn
        self.H = self.f_rnn.hidden_weight.shape[0]
        self.output_weight = np.random.randn(output_dim, 2*self.H)
        self.output_bias = np.random.randn(output_dim,)
        
        
    def forward(self, X):
        self.f_rnn.forward(X)
        self.b_rnn.forward(X[::-1])

        f_T = self.f_rnn.hidden_states[-1]
        b_T = self.b_rnn.hidden_states[-1]
        
        self.final_hidden_state = np.concatenate([f_T, b_T], axis= 0)
        logits = self.output_weight @ self.final_hidden_state + self.output_bias 
        probs = softmax(logits)
        return probs
    
    def backwards(self, X, y, probs):
        delta_out = probs - y
        grad_output = np.outer(delta_out, self.final_hidden_state)
        grad_output_bias = delta_out
        
        delta_h = self.output_weight.T @ delta_out
        delta_f = delta_h[:self.H]
        delta_b = delta_h[self.H:]
        
        f_grads = self.f_rnn.backwards(delta_f, X)
        b_grads = self.b_rnn.backwards(delta_b, X[::-1])
        
        return f_grads, b_grads, grad_output, grad_output_bias

    
    def update(self, grads, eta):
        f_grads, b_grads, weight_grad, bias_grad = grads
        self.output_weight -= eta * weight_grad
        self.output_bias -= eta * bias_grad
        self.f_rnn.update(f_grads, eta)
        self.b_rnn.update(b_grads, eta)
        
    def train(self, X, y, epochs, eta):
        for e in range(epochs):
            probs = self.forward(X)
            loss = ce(y, probs)
            grads = self.backwards(X, y, probs)
            self.update(grads, eta)
            print(f'Epoch {e} | CE Loss: {loss}')
    
    def predict(self, X, sentiments):
        probs = self.forward(X)
        sample = np.random.choice(probs.shape[0], p= probs)
        sentiment = sentiments[sample]
        print(f'For this sequence the chosen sentiment is {sentiment} with probability {probs[sample] * 100}')
            
def ce(y, probs):
    loss = -(y * np.log(probs)).sum()
    return loss       
        
def softmax(logits):
    # prevent overflow
    logits = logits - logits.max()
    
    probs = np.exp(logits) / np.exp(logits).sum()
    return probs    

class seq2seq:
    # encoder -> context vector -> decoder
    def __init__(self, encoder: Encoder, decoder: vec2seq) -> None:
        self.encoder = encoder
        self.decoder = decoder
        decoder_D = self.decoder.w_xh.shape[1] - self.decoder.C
        self.W_c = np.random.randn(decoder_D , self.encoder.hidden_units)
        self.b_c = np.random.randn(decoder_D)
    
    def forward(self, X, y):
        # X: T, D sequence of length T with D dim
        # y: T', target labels in token form
        h_T = self.encoder.forward(X)
        c = self.W_c @ h_T + self.b_c
        self.decoder.forward(c, y)
        return 
    
    def backwards(self, X, y):
        decoder_gradients, encoder_delta = self.decoder.backwards(y)
        
        grad_W_c = np.outer(encoder_delta, self.encoder.hidden_states[-1]) 
        grad_b_c = encoder_delta
        
        encoder_delta = self.W_c.T @ encoder_delta # H_exH_d x H_d,
        
        encoder_gradients = self.encoder.backwards(encoder_delta, X)
        
        
        return encoder_gradients, decoder_gradients, grad_W_c, grad_b_c
        
    def update(self, encoder_gradients, decoder_gradients, proj_grads, eta):
        grad_W_c, grad_b_c = proj_grads
        self.W_c -= eta * grad_W_c
        self.b_c -= eta * grad_b_c
        self.encoder.update(encoder_gradients, eta)
        self.decoder.update(decoder_gradients, eta)
        
    def train(self, X, y, eta, epochs):
        for e in range(epochs):
            self.forward(X, y)
            encoder_grads, decoder_grads, *proj_grads = self.backwards(X, y)
            loss = ce(self.decoder.y, self.decoder.P)
            self.update(encoder_grads, decoder_grads, proj_grads, eta)
            print(f'Epoch {e} | Loss: {loss}')
    