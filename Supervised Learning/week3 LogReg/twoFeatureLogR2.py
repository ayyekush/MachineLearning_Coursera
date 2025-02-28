# Only for fictional purposes. Two feature regression where the 
       # decision boundary is not a straight line.

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

# Generate sample data with two features
np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Fit logistic regression model
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_poly, y)

# Plotting 3D decision boundary
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(X[:, 0], X[:, 1], y, c=y, marker='o')

# Plotting the 3D decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_values = np.linspace(x_min, x_max, 100)
y_values = np.linspace(y_min, y_max, 100)
xx, yy = np.meshgrid(x_values, y_values)
zz = -(clf.intercept_[0] + clf.coef_[0][1]*yy + clf.coef_[0][2]*xx +
       clf.coef_[0][3]*xx*yy + clf.coef_[0][4]*xx**2 + clf.coef_[0][5]*yy**2)

ax.plot_surface(xx, yy, zz, alpha=0.5)

# Plotting the decision boundary
x_values = np.linspace(x_min, x_max, 100)
y_values = np.linspace(y_min, y_max, 100)
X_grid, Y_grid = np.meshgrid(x_values, y_values)
Z_decision_boundary = (clf.predict(poly.fit_transform(np.c_[X_grid.ravel(), Y_grid.ravel()]))).reshape(X_grid.shape)
ax.contourf(X_grid, Y_grid, Z_decision_boundary, zdir='z', offset=0.5, alpha=0.2)

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Probability')
ax.set_title('Combined 3D Decision Boundary')

plt.show()