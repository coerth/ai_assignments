#%%

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#%%

np.set_printoptions(precision=3)

#%%

x = np.arange(5)
y = np.array([2.1, 3.5, 3.9, 5.3, 5.6])
# example: y ≈ x + 2

#%%

plt.scatter(x, y)
plt.plot(x, y)
plt.plot(x, np.array([2, 3, 4, 5, 6]))
plt.show()

#%%

class Linear:
    def __init__(self, theta: np.array = np.array([0, 0])):
        self.theta = theta.astype('float64')
        
    def __call__(self, x: np.array) -> np.array:
        return self.theta[1] * x + self.theta[0]
    
    def partial_diff(self, x: np.array, y: np.array) -> np.array:
        dJ_dtheta0 = -(1/len(y)) * np.sum( self(x) - y )
        dJ_dtheta1 = -(1/len(y)) * np.sum( (self(x) - y) * x )
        return np.array([dJ_dtheta0, dJ_dtheta1])
    
    def delta(self, delta: np.array) -> None:
        self.theta += delta
        
    def loss(self, x: np.float64, y: np.float64) -> np.float64:
        # the error of a single sample x, ground truth value y
        return (self(x) - y)**2
    
    def cost(self, x: np.array, y: np.array) -> np.float64:
        # total cost of all samples x, ground truth y
        return (1/(2 * len(y))) * np.sum( (self(x) - y)**2 )
    
#%%
    
    

def gradient_descent(x: np.array, y: np.array, model, iterations: int, learning_rate: float):
    print('theta\t\tcost\tpartial_diff')
    for i in range(iterations):
        model.delta(learning_rate * model.partial_diff(x, y))
        if i % (iterations / 10) == 0:
            print(model.theta, end='\t')
            print(f'{model.cost(x, y):.3f}', end='\t')
            print(model.partial_diff(x, y))
    return model.theta

#%%



theta = gradient_descent(x, y, Linear(), 8000, 0.001)
print('--')
print(theta)


# %%



plt.scatter(x, y)
plt.plot(x, y)
plt.plot(x, [theta[1] * j + theta[0] for j in x])


# %%

xm, ym = np.linspace(1, 3, 50), np.linspace(0, 2, 50)

def contour_surface(xm, ym, x, y, model):
    z = np.zeros([len(xm), len(ym)])
    for ii, i in enumerate(xm):
        for ij, j in enumerate(ym):
            z[ii, ij] = model(theta=np.array([i, j])).cost(x, y)
    return z

Linear(theta=np.array([2, 1])).cost(x, y)


# %%

plt.contourf(contour_surface(xm, ym, x, y, Linear), levels=255, cmap='Reds')
# %%

plt.contour(contour_surface(xm, ym, x, y, Linear), levels=255, cmap='Reds')


# %%

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(xm, ym, contour_surface(xm, ym, x, y, Linear))


# %%
