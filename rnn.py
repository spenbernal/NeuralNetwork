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
    
    def update(self, gradients):
        # gradient descent
        grad_input, grad_hidden, grad_hidden_bias, grad_output, grad_output_bias = gradients
        
        self.input_weight -= self.eta * grad_input
        self.hidden_weight -= self.eta * grad_hidden
        self.output_weight -= self.eta * grad_output
        self.bias_hidden -= self.eta * grad_hidden_bias
        self.bias_output -= self.eta * grad_output_bias
        
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

