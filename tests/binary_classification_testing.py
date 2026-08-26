import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sklearn.datasets import load_breast_cancer
import numpy as np
import linearmodels.LogisticRegression as logreg
import matplotlib.pyplot as plt
from pathlib import Path

test_dir = Path(__file__).parent
print('Parent Directory', test_dir)
result_dir = test_dir / 'regression_results'
print('Result Directory', result_dir)
result_dir.mkdir(exist_ok= True)

data = load_breast_cancer()
X = data.data #type: ignore # N,D design matrix
y = data.target #type: ignore #N, target vector

# shuffle and create train and test datasets
test_size = 0.2
N = X.shape[0]
N_test = int(test_size * N)
rand_indices = np.random.permutation(N)
test_idx = rand_indices[:N_test]
train_idx = rand_indices[N_test:]
X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

# preprocess data (standardization)
mu = X_train.mean(axis= 0)
std = X_train.std(axis= 0)
X_train_scaled = (X_train - mu) / std
X_test_scaled = (X_test - mu) / std

# initialize models
model = logreg.LogisticRegression()

# initialize training parameters
eta = 0.01
epochs = 100

# train models
# first order optimization
for e in range(epochs):
    probs = model.forward(X_train_scaled) # N,
    loss = -(y_train * np.log(probs) + (1-y_train) * np.log(1 - probs)).mean()
    print(f'Epoch {e+1} | Binary Cross Entropy: {loss}')
    delta = probs - loss
    model.backwards(delta)
    model.update(eta)
# print weights and bias
print('---------- First order gradient method ----------')
for d in range(model.weights.shape[0]):
    feature = data.feature_names[d] # type: ignore
    print(f'Feature: {feature} | Weight: {model.weights[d]}')
print(f'Bias: {model.bias}')

# evaluate on testing data
probs = model.forward(X_test)
loss = -(y_test * np.log(probs) + (1-y_test) * np.log(1 - probs)).mean()
y_preds = (probs >= 0.5).astype(int)
accuracy = np.mean(y_test == y_preds)
TP = np.sum((y_test == 1) & (y_preds == 1))
FP = np.sum((y_test == 0) & (y_preds == 1))
precision = TP / (TP + FP)
print(f'Loss on test set (BCE): {loss} | Accuracy: {accuracy} | Precision: {precision}')



# second order optimization
model.IRLS(X_train_scaled, y_train, epochs)

# print weights and bias
print('---------- Second order gradient method ----------')
for d in range(model.weights.shape[0]):
    feature = data.feature_names[d] # type: ignore
    print(f'Feature: {feature} | Weight: {model.weights[d]}')
print(f'Bias: {model.bias}')

# evaluate on testing data
logits = X_test @ model.weights
probs = 1 / (1 + np.exp(-logits))
loss = -(y_test * np.log(probs) + (1-y_test) * np.log(1 - probs)).mean()
y_preds = (probs >= 0.5).astype(int)
accuracy = np.mean(y_test == y_preds).mean()
TP = np.sum((y_test == 1) & (y_preds == 1))
FP = np.sum((y_test == 0) & (y_preds == 1))
precision = TP / (TP + FP)
print(f'Loss on test set (BCE): {loss} | Accuracy: {accuracy} | Precision: {precision}')

