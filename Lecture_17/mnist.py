#%%

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
#%%

mnist = fetch_openml('mnist_784', version=1)

#%%
# make a decision tree classifier
from sklearn.tree import DecisionTreeClassifier

X, y = mnist['data'], mnist['target']

#split the data into training and testing and validation
X_train, X_test, X_val = X[:50000], X[50000:60000], X[60000:70000]
y_train, y_test, y_val = y[:50000], y[50000:60000], y[60000:70000]

#%%

# train the model
start = time.time()
clf = DecisionTreeClassifier()
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

# visualize the decision tree
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 20))
plot_tree(clf, filled=True)
plt.show()

#%%

#save the tree
plt.savefig('mnist_tree.png')

# %%

# # tune the model using validation set
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'max_depth': [10, 20, 30, 40],
#     'min_samples_split': [2, 5, 10, 15],
#     'min_samples_leaf': [1, 2, 5, 10]
# }

# grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_val, y_val)

# # %%

# # get the best parameters
# best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')

# # %%

# # train the model with the best parameters
# start = time.time()
# clf = DecisionTreeClassifier(**best_params)
# clf.fit(X_train, y_train)
# end = time.time()

# # %%

# # test the model
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
# print(f'Time taken: {end - start}')

# %%

