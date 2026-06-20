import numpy as np
class seq2vec:
    # Time series forecast of fish price data
    def __init__(self, M, N, hidden_units, output_dim) -> None:
        
        self.M = M
        self.N = N
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.input_weight = np.random.randn(self.hidden_units, self.N)
        self.hidden_weight = np.random.randn(self.hidden_units, self.hidden_units)
        self.output_weight = np.random.randn(self.output_dim, self.hidden_units)
        self.bias_hidden = np.random.randn(self.hidden_units)
        self.bias_output = np.random.randn(self.output_dim,)
        
    def forward(self, X):
        self.hidden_states = [np.zeros(self.hidden_units)]
        for i in range(self.M):
            x = X[i] 
            hidden_state = ReLU(self.input_weight @ x 
                                + self.hidden_weight @ np.array(self.hidden_states[-1]) 
                                + self.bias_hidden)
            
            self.hidden_states.append(hidden_state)
        
        y_hat = self.output_weight @ self.hidden_states[-1] + self.bias_output
        return y_hat
    
    def backwards(self, y_hat, y, X):
        delta_out = (y_hat - y) 
        grad_output = np.outer(delta_out, self.hidden_states[-1]) #(output,)x(hidden_dim,) = (output, hidden)
        grad_output_bias = delta_out 
        
        delta = (self.output_weight.T @ delta_out) * np.where(self.hidden_states[-1] > 0, 1, 0) #(hidden,)
        grad_input = np.zeros_like(self.input_weight)
        grad_hidden = np.zeros_like(self.hidden_weight)
        grad_hidden_bias = np.zeros_like(self.bias_hidden)
        
        for t in range(self.M, 0, -1):
            grad_input += np.outer(delta, X[t-1]) 
            grad_hidden += np.outer(delta, self.hidden_states[t-1])
            grad_hidden_bias += delta
            
            delta = (self.hidden_weight.T @ delta) * np.where(self.hidden_states[t-1] > 0, 1, 0)        
            
        return grad_input, grad_hidden, grad_hidden_bias, grad_output, grad_output_bias
    
    def backwards_hidden(self, delta_h, X):  
        delta = (delta_h) * np.where(self.hidden_states[-1] > 0, 1, 0) #(hidden,)
        grad_input = np.zeros_like(self.input_weight)
        grad_hidden = np.zeros_like(self.hidden_weight)
        grad_hidden_bias = np.zeros_like(self.bias_hidden)
        
        for t in range(self.M, 0, -1):
            grad_input += np.outer(delta, X[t-1]) 
            grad_hidden += np.outer(delta, self.hidden_states[t-1])
            grad_hidden_bias += delta
            
            delta = (self.hidden_weight.T @ delta) * np.where(self.hidden_states[t-1] > 0, 1, 0)        
            
        return grad_input, grad_hidden, grad_hidden_bias
    
    
    def update(self, gradients):
        # gradient descent
        grad_input, grad_hidden, grad_hidden_bias, grad_output, grad_output_bias = gradients
        
        self.input_weight -= self.eta * grad_input
        self.hidden_weight -= self.eta * grad_hidden
        self.output_weight -= self.eta * grad_output
        self.bias_hidden -= self.eta * grad_hidden_bias
        self.bias_output -= self.eta * grad_output_bias
    
    def update_hidden(self, gradients):
        grad_input, grad_hidden, grad_hidden_bias = gradients
        
        self.input_weight -= self.eta * grad_input
        self.hidden_weight -= self.eta * grad_hidden
        self.bias_hidden -= self.eta * grad_hidden_bias
        
        
    def train(self, X, y, epochs, eta):
        self.eta = eta
        for _ in range(epochs):
            y_hat = self.forward(X)
            loss = MSE(y, y_hat)
            gradients = self.backwards(y_hat, y, X)
            self.update(gradients)
            print(f'Training Loss: {loss.round(5)}')

    def predict(self, X):
        prediction = self.forward(X)
        print(f'The prediction for the next price is ${prediction.round(3)}')
        
def ReLU(x):
    return np.maximum(0,x)

def MSE(y, y_hat):
    return 1/2 * (y_hat - y).sum()**2

class seq2vecBidirectional:
    # bidirectional RNN for sentiment classification
    def __init__(self, f_rnn: seq2vec, b_rnn: seq2vec, output_dim) -> None:
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
    
    def backwards(self, probs, y, X):
        delta_out = probs - y
        grad_output = np.outer(delta_out, self.final_hidden_state)
        grad_output_bias = delta_out
        
        delta_h = self.output_weight.T @ delta_out
        delta_f = delta_h[:self.H]
        delta_b = delta_h[self.H:]
        
        f_grads = self.f_rnn.backwards_hidden(delta_f, X)
        b_grads = self.b_rnn.backwards_hidden(delta_b, X[::-1])
        
        
        return f_grads, b_grads, grad_output, grad_output_bias

    
    def update(self, grads, eta):
        f_grads, b_grads, weight_grad, bias_grad = grads
        self.output_weight -= eta * weight_grad
        self.output_bias -= eta * bias_grad
        self.f_rnn.update_hidden(f_grads)
        self.b_rnn.update_hidden(b_grads)
        
    def train(self, X, y, epochs, eta):
        for e in range(epochs):
            probs = self.forward(X)
            loss = ce(y, probs)
            grads = self.backwards(probs, y, X)
            self.update(grads, eta)
            print(f'Epoch {e} | CE Loss: {loss}')
    
    def predict(self, X, sentiments):
        probs = self.forward(X)
        sentiment = sentiments[np.argmax(probs)]
        print(f'For this sequence the chosen sentiment is {sentiment} with probability {probs.max() * 100}')
            
def ce(y, probs):
    loss = -(y * np.log(probs)).sum()
    return loss       
        
def softmax(logits):
    # prevent overflow
    logits = logits - logits.max()
    
    probs = np.exp(logits) / np.exp(logits).sum()
    return probs        
    
    
