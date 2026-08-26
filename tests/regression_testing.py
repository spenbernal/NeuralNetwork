import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sklearn.datasets import fetch_california_housing
import numpy as np
import linearmodels.LinearRegression as reg
import matplotlib.pyplot as plt
from pathlib import Path

test_dir = Path(__file__).parent
print('Parent Directory', test_dir)
result_dir = test_dir / 'regression_results'
print('Result Directory', result_dir)
result_dir.mkdir(exist_ok= True)


housing = fetch_california_housing()
X = housing.data # type: ignore # N, D 
y = housing.target # type: ignore shape:  N

# regularizers
lambdas = [0.1, 1, 100]
ridge_reg = {} # lambda: regression() (key:val)
lasso_reg = {}

# get random mask of data
test_size = 0.2
N = X.shape[0]
test_set = np.random.choice([False, True], size= N, p= [1-test_size, test_size]) # N, 

X_test = X[test_set]
y_test = y[test_set]

X_train = X[~test_set]
y_train = y[~test_set]

# standardize training and test data
mu = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mu) / std
X_test = (X_test - mu) / std
N = X_train.shape[0]
M = X_test.shape[0]
X_train = np.column_stack([np.ones(N), X_train])
X_test = np.column_stack([np.ones(M), X_test])

# initialize models
linear_reg = reg.LinearRegression()
for lam in lambdas:
    ridge_reg[lam] = reg.RidgeRegression(lam)
    lasso_reg[lam] = reg.LassoRegression(lam)
    
# train models
linear_reg.fit(X_train, y_train)
for lam in lambdas:
    ridge_reg[lam].fit(X_train, y_train)
    lasso_reg[lam].fit(X_train, y_train)

# evaluate models
print('---------- Linear Regression ----------')
print(f'Loss (MSE): {linear_reg.eval(X_test, y_test)[0]} \
      | R2 Score: {linear_reg.eval(X_test, y_test)[1]}')
print('***************')
      
for lam in lambdas:
    print(f'---------- Ridge Regression (lambda = {lam}) ----------')
    print(f'Loss (MSE): {ridge_reg[lam].eval(X_test, y_test)[0]} \
      | R2 Score: {ridge_reg[lam].eval(X_test, y_test)[1]}')
    print('***************')
    
    print(f'---------- Lasso Regression (lambda = {lam}) ----------')
    print(f'Loss (MSE): {lasso_reg[lam].eval(X_test, y_test)[0]} \
          | R2 Score: {lasso_reg[lam].eval(X_test, y_test)[1]}')
    print('***************')

# plot models
# plotting wrt 'MedInc' 
fig, ax = plt.subplots(1, 1)
fig.suptitle('Linear Regression')
ax.scatter(X_test[:, 1], y_test, label='test')
x = np.linspace(X_test[:,1].min(), X_test[:,1].max(), 200)
bias = linear_reg.weights[0]
y = bias + x * linear_reg.weights[1]
ax.plot(x, y, color='red', label='reg') 
ax.grid(True)
ax.legend()
ax.set_xlabel('MedInc')

fig.tight_layout()
fig.savefig(result_dir / 'linear_plots.png', dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize= (15,7))
fig.suptitle('Ridge Regression')
for i in range(3):
    ax[i].set_title(f'Ridge Regression ($\lambda = {lambdas[i]}$)')
    ax[i].scatter(X_test[:, 1], y_test, label='test')
    b = ridge_reg[lambdas[i]].weights[0]
    y = ridge_reg[lambdas[i]].weights[1] * X_test[:,1] + b
    ax[i].plot(X_test[:, 1], y, label='reg', color='pink')
    ax[i].grid(True)
    ax[i].set_xlabel('MedInc')
    ax[i].legend()

fig.tight_layout()
fig.savefig(result_dir / 'ridge_plots.png', dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize= (15,7))
fig.suptitle('Lasso Regression')
for i in range(3):
    ax[i].set_title(f'Lasso Regression ($\lambda = {lambdas[i]}$)')
    ax[i].scatter(X_test[:, 1], y_test, label='test')
    b = lasso_reg[lambdas[i]].weights[0]
    y = lasso_reg[lambdas[i]].weights[1] * X_test[:,1] + b
    ax[i].plot(X_test[:, 1], y, label='reg', color='orange')
    ax[i].grid(True)
    ax[i].set_xlabel('MedInc')
    ax[i].legend()


fig.tight_layout()
fig.savefig(result_dir / 'lasso_plots.png', dpi=300, bbox_inches="tight")
plt.close(fig)