# ITP we visualise value of w and b vs f in 3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the compute_cost function
def compute_cost(x, y, w, b):
    m = len(x)
    cost = np.sum((w*x+b - y) ** 2) / (2 * m)
    return cost

# Data
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250,300,480,430,630,730])

# Define ranges for w and b values
w_values = np.linspace(100, 400, 100) #linspace(start,end,no of points)
#creates a range in tensor of 100 val between start and end 
b_values = np.linspace(0, 200, 100)

# Create meshgrid for w and b values
W, B = np.meshgrid(w_values, b_values)

# Compute cost for each combination of w and b
# l1=[]
# l2=[]
# for i in w_values:
#     for j in b_values:
#         l2.append(compute_cost(x_train,y_train,i,j))
#     l1.append(l2)
#     l2=[]
# cost_values=np.array(l1)
cost_values=np.array([[compute_cost(x_train,y_train,w,b) for b in b_values] for w in w_values])
# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, cost_values, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost')
ax.set_title('Cost Function in 3D')
plt.show()