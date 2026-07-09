import numpy as np

class SGD:
    def __init__(self, eta) -> None:
        self.eta = eta
    
    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'W'):
                layer.update(self.eta)