# In this we plot logical regression for two features and the 3d 
    # sigmoid graph to see how it works

import matplotlib.pyplot as plt
import numpy as np

# Sample data with two features
x1 = np.array([-4, -3, -5, 2, 3, 4])
x2 = np.array([1, 2, -3, -4, 2, 1])
y = np.array([0, 0, 1, 1, 1, 1])

# Normalize input data
# the data is scaled so youre not gonna see (-4,1) you fool
x1 = (x1 - np.mean(x1)) / np.std(x1)
x2 = (x2 - np.mean(x2)) / np.std(x2)
# try unscaling and see why scaling is preferred
m = len(x1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(x1, x2, y, epochs, learning_rate):
    w1 = 0
    w2 = 0
    b = 0
    for epoch in range(epochs):
        z = w1 * x1 + w2 * x2 + b
        h = sigmoid(z)
        dw1 = 1/m * np.sum((h-y)*x1)
        dw2 = 1/m * np.sum((h-y)*x2)
        db = 1/m * np.sum(h - y)
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        b -= learning_rate * db
    return w1, w2, b

# THIS THE THE 2D DECISION BOUNDARY GRAPH
w1, w2, b = logistic_regression(x1, x2, y, epochs=10000, learning_rate=0.01)

x1_values = np.linspace(-2, 2, 50)
x2_values = np.linspace(-2, 2, 50)
X1, X2 = np.meshgrid(x1_values, x2_values)
Z = sigmoid(w1 * X1 + w2 * X2 + b)

# Plotting the decision boundary with sigmoid function
fig = plt.figure(figsize=(12, 6))

# Subplot 1: 3D sigmoid function with decision boundary plane
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X1, X2, Z,cmap='viridis', alpha=0.5)
x_values = np.linspace(-2, 2, 100)
y_values = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_values, y_values)
Z_decision_boundary = 0.5 * np.ones_like(X)
ax1.plot_surface(X, Y, Z_decision_boundary, color='gray', alpha=0.3)
x_decision_boundary = np.linspace(-2, 2, 100)
y_decision_boundary = -(w1 * x_decision_boundary + b) / w2
ax1.plot(x_decision_boundary, y_decision_boundary, 0.5, color='red', label='Decision Boundary')
ax1.scatter(x1, x2, y, c=y)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Probability')
ax1.set_title('3D Sigmoid Function with Decision Boundary')

# Subplot 2: Decision boundary curve
ax2 = fig.add_subplot(122)
x_values = np.linspace(-2, 2, 100)
y_values = -(w1 * x_values + b) / w2
ax2.plot(x_values, y_values, label='Decision Boundary')
ax2.scatter(x1, x2, c=y)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('Decision Boundary Curve')

plt.legend()
plt.tight_layout()
plt.show()
