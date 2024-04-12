#%%

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
#%%

mnist = fetch_openml('mnist_784', version=1)

#%%

# make a random forest classifier
from sklearn.ensemble import RandomForestClassifier

X, y = mnist['data'], mnist['target']

#split the data into training and testing and validation
X_train, X_test, X_val = X[:50000], X[50000:60000], X[60000:70000]
y_train, y_test, y_val = y[:50000], y[50000:60000], y[60000:70000]

#%%
# train the model
start = time.time()
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
end = time.time()

# %%

# test the model
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print(f'Time taken: {end - start}')

# %%

# use entropy instead of gini

# train the model
start = time.time()
clf = RandomForestClassifier(criterion='entropy')
clf.fit(X_train, y_train)
end = time.time()

# %%