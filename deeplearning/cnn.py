import deeplearning.layers as l
import deeplearning.nn as nn
class CNN:
    # A simple CNN for identifying hand drawn digits
    def __init__(self, conv_layers: list[l.Layer], NN: nn.Sequential) -> None:
        self.conv_layers = conv_layers
        self.NN = NN
    
    def forward(self, X):
        # N, C_in, H, W
        self.Z = X
        for layer in self.conv_layers:
            self.Z = layer.forward(self.Z)
        
        logits = self.NN.forward(self.Z)
        return logits
    
    def backwards(self, delta):
        delta = self.NN.backwards(delta)
        for layer in reversed(self.conv_layers):
            delta = layer.backwards(delta)
        return delta
    
    def update(self, eta):
        for layer in self.conv_layers:
            layer.update(eta)
        self.NN.update(eta)

