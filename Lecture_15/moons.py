#%%
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import utils
#%%
np.random.seed(0)

m = 1_000
split_train = int(m * 0.7)
split_val = int(m * 0.15 + split_train)
split_test = int(m * 0.15 + split_val)

X, y = datasets.make_moons(
    n_samples=m, 
    noise=0.1, 
    random_state=0
)

X_train, y_train = X[:split_train], y[:split_train]
X_val, y_val = X[split_train:split_val], y[split_train:split_val]
X_test, y_test = X[split_val:split_test], y[split_val:split_test]
#%%