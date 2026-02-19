import numpy as np
# Binary Classification
# loss: BCE
# activation: sigmoid
class NN:
    # Initialization
    def __init__(self, sigmoid, reLu, loss_function, input_dim, output_dim, lr):
        #input layer, 10 Neuron Hidden Layer, 3 Neuron Hidden Layer, Output Layer
        # num of features
        self.input_dim = input_dim
        # num of outputs (2 since binary)
        self.output_dim = output_dim
        # neurons per layer
        self.layer_dims = [input_dim, 10, 3, output_dim]

        # weights and biases for each layer
        self.weights = []
        self.biases = []
        # set up activation and loss functions
        self.sigmoid = sigmoid
        self.reLu = reLu
        self.loss_function = loss_function
        
        self.lr = lr
        
        #setting up weights and biases in layers
        n = len(self.layer_dims)
        for i in range(n - 1):
            #(input,10) , (10,3), (3,1)
            self.weights.append(np.random.randn(self.layer_dims[i], self.layer_dims[i+1]))
            self.biases.append(np.random.randn(self.layer_dims[i+1]))
    
    def printModel(self):
        print('********** Model Architecture **********')
        print(f'# of Hidden Layers: {len(self.layer_dims) - 2}')
        print(f'Input Dim: {self.input_dim}')
        for i in range(1, len(self.layer_dims) - 1): 
            print(f'Neurons in Hidden Layer {i}: {self.layer_dims[i]}')  
        print(f'Output Dim: {self.output_dim}')
        print(f'Loss Function: {self.loss_function.__name__}')  #type: ignore
    
    # Forward Pass     
    def forward(self, X):
        # X = n_samples, features
        a = X
        memo = {} #stores input, linear output, and activated output
        n_hidden = len(self.weights) - 1
        for i in range(n_hidden):
            input = a
            z = np.dot(a, self.weights[i]) + self.biases[i] 
            a = self.reLu(z)
            memo[i] = (input, z, a) # input, linear output, activated output
            
        out = np.dot(a, self.weights[-1]) + self.biases[-1]
        activ_out = self.sigmoid(out)
        k = len(memo)
        memo[k] = (a, out, activ_out)

        return activ_out, memo

    
    def backPropogation(self, y, history):
        # Loss = loss of forward run
        # history = memo from forward run
        N = y.shape[0]
        grad_ws = {}
        grad_bs = {}
        last_layer = len(self.weights) - 1
        input, _, y_hat = history[last_layer] #input = (10,1) ,_ = (3,1), y_hat = (1,1)
        delta = y_hat - y
        grad_ws[last_layer] = input.T @ delta / N # delta = (N,1), input = (N, 3)
        grad_bs[last_layer] = np.sum(delta, axis= 0) / N # N, 1 -> N,3 -> N,10
        n = len(self.weights) - 1
        for i in range(n - 1, -1, -1):
            input, z, a_z = history[i]
            delta = (delta @ self.weights[i+1].T) * deriv_reLu(z) #N,1 x 1,3
            
            grad_ws[i] = input.T @ delta / N
            grad_bs[i] = np.sum(delta, axis= 0) / N
        
        return grad_ws, grad_bs
    
    # Gradient Descent - Parameter Optimization
    def gd(self, grad_ws, grad_bs):
        n = len(self.weights)
        for i in range(n):
            self.weights[i] -= self.lr * grad_ws[i]
            self.biases[i] -= self.lr * grad_bs[i] 
        return    
    
    def accuracy(self, y_true, y_pred):
        y_pred_class = (y_pred >= 0.5).astype(int)
        return np.mean(y_pred_class.flatten() == y_true.flatten())
        
    def train(self, X_train, y_train, batch_size= 1, epochs=1000):
        for e in range(epochs):
            # forward, loss, backprop, update
            y_pred, history = self.forward(X_train)
            loss = self.loss_function(y_train, y_pred) # type:ignore
            acc = self.accuracy(y_train, y_pred)
            grad_ws, grad_bs = self.backPropogation(y_train, history)
            self.gd(grad_ws, grad_bs)
            
            print(f'Epoch: {e + 1} | Loss : {loss:.5f} | Accuracy : {acc:.5f}')
            
        print(f'{epochs} Done!')
        return
    
    def eval(self, X_test, y_test):
        y_pred, _ = self.forward(X_test)
        loss = self.loss_function(y_test, y_pred)
        acc = self.accuracy(y_test, y_pred)
        print(f'Loss: {loss:.5f} | Accuracy: {acc:5f}')
        return

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def reLu(z):
    return np.maximum(0, z)   

def deriv_reLu(z):
    return np.where(z > 0, 1, 0)  

def BinaryCrossEntropyLoss(y, y_hat):
    # y = (n_samples, 1)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    loss = -(y * np.log(y_hat) + (1 - y) * np.log(1- y_hat))
    return np.mean(loss)



    
   