import numpy as np
class Linear:
    def __init__(self, N, o) -> None:
        # N: dim of vector
        # o: output dim
        self.N = N
        self.o = o
        self.W = np.random.randn(o, N) # N, o
        self.b = np.random.randn(o)
    
    def forward(self, x):
        # X: N, input vector
        self.x = x # N
        a = x @ self.W.T + self.b 
        return a
    
    def backwards(self, delta): 
        # delta: N, error from future layer
        self.grad_W = np.outer(delta, self.x)
        self.grad_b = delta
        
        # update delta
        delta = self.W.T @ delta
        return delta
    
    def update(self, eta):
        self.W -= eta * self.grad_W
        self.b -= eta * self.grad_b
        return 
class ReLU:
    def forward(self, logits):
        # logits: 
        self.logits = logits
        return np.maximum(0, logits)
    
    def backwards(self, delta):
        delta = delta * np.where(self.logits > 0, 1, 0)
        return delta
    
    def update(self, eta):
        pass

class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        self.y = y
        self.probs = np.exp(logits) / np.exp(logits).sum()
        loss = -(y * np.log(self.probs)).sum()
        return loss
    
    def backwards(self):
        # output gradient
        return self.probs - self.y
    
    def update(self, eta):
        pass
    
class Dropout:
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
class Sequential:
    # A classic feedforward NN module
    # training and evaluation are task dependent so they live outside the module
    def __init__(self, layers) -> None:
        # layers: list of layers     
        self.layers = layers
        self.training = True
    
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
        
    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        
        return X
    
    def backwards(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backwards(delta) 
        return delta
        
    def update(self, eta):
        for layer in self.layers:
            layer.update(eta)
        return

class ResNet:
    # A residual neural network
    def __init__(self, nn: Sequential, proj: Linear) -> None:
        self.proj = proj
        self.nn = nn
    
    def forward(self, X):
        # want to feed X into the last layer
        # assume layer has same output shape as X
        logits = self.nn.forward(X)
        logits = self.proj.forward(logits) + X
        return logits 

    def backwards(self, delta):
        delta_skip = delta    
        
        delta_main = self.proj.backwards(delta)
        delta_main = self.nn.backwards(delta_main)
        delta_x = delta_main + delta_skip
        return delta_x
    
    def update(self, eta):
        self.proj.update(eta)
        self.nn.update(eta)
        return 
        
        
    




    
   