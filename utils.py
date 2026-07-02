import numpy as np

def softmax(logits: np.ndarray):
    return np.exp(logits) / np.exp(logits).sum()
    