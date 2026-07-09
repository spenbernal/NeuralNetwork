import numpy as np
from abc import ABC, abstractmethod   
class Criterion(ABC):
    
    @abstractmethod
    def forward(self, logits, y):
        pass
    
    @abstractmethod
    def backwards(self):
        pass
    
class SoftmaxCrossEntropy(Criterion):
    def forward(self, logits, y):
        self.y = y
        self.probs = np.exp(logits) / np.exp(logits).sum()
        loss = -(y * np.log(self.probs)).sum()
        return loss
    
    def backwards(self):
        # output gradient
        return self.probs - self.y
    
class BinaryCrossEntropy(Criterion):
    def forward(self, logit, y):
        # logit: scalar
        self.y = y
        self.prob = 1 / (1 + np.exp(-logit))
        loss = -(y * np.log(self.prob) + (1 - y)* np.log(1 - self.prob)).sum()
        return loss
        
    def backwards(self):
        return self.prob - self.y
    
class MSE(Criterion):
    def forward(self, y_hat, y):
        self.y = y
        self.y_hat = y_hat
        loss = 0.5 * ((y_hat - y)**2).mean()
        return loss
    
    def backwards(self):
        return self.y_hat - self.y

        
