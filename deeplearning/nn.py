import deeplearning.layers as l
import numpy as np
class Sequential:
    # Simple chain neural network 
    def __init__(self, layers: list[l.Layer]) -> None:
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
    
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

class ResNet:
    # Example of a residual neural network
    def __init__(self, nn: Sequential, proj: l.Linear) -> None:
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
        
        
    




    
   