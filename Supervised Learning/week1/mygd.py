# ITP I try to make gd for multi-variable linear regression

import numpy as np
import matplotlib.pyplot as plt

x1=np.array([2,3,4,5,8,6])
x2=np.array([3,4,1,2,7,8])
x3=np.array([1,2,4,5,2,2])
y=np.array([100,350,370,420,870,750])
w1=1
w2=2
w3=0
b=0
m=x1.shape[0]
learning_rate=0.001
w=np.array([w1,w2,w3])
# fwb=w1x1[i]+w2x2[i]+w3x3[i];
# J=(1/2m)sum(w1x1+w2x2+w3x3+b  -y)**2
epoch=100000;
for i in range(epoch):
    dw1=(1/m)*np.sum((w1*x1+w2*x2+w3*x3+b-y)*x1)
    #np.sum(w1*x)==(w1*x[0])+(w1*x[1])+...
    dw2=(1/m)*np.sum((w1*x1+w2*x2+w3*x3+b-y)*x2)
    dw3=(1/m)*np.sum((w1*x1+w2*x2+w3*x3+b-y)*x3)
    db=(1/m)*np.sum(w1*x1+w2*x2+w3*x3+b-y)
    w1-=learning_rate*dw1
    w2-=learning_rate*dw2 
    w3-=learning_rate*dw3
print("w1,w2,w3",w1,w2,w3)
# yp=np.array(w1*x1+w2*x2+w3*x3+b)
# plt.scatter(x1,y)
# plt.plot(x1,yp)
# plt.show()

y_pred=np.array(w1*x1+w2+x2+w3+b)
# Plotting the predicted values against the actual values
# fid,(subFig1,subFig2)=plt.subplot(1,2)
# plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue',label='Deviations')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red',label='x=y')  # Plotting the y = x line
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual vs Predicted values')
plt.grid(True)
plt.legend()
plt.show()

# Plotting the regression plane
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Creating meshgrid for x1 and x2
x1_mesh, x2_mesh = np.meshgrid(np.linspace(min(x1), max(x1), 10), np.linspace(min(x2), max(x2), 10))

# Calculating corresponding x3 values for the plane
x3_mesh = -(w1*x1_mesh + w2*x2_mesh + b) / w3

# Plotting the regression plane
ax.plot_surface(x1_mesh, x2_mesh, x3_mesh, alpha=0.5)

# Plotting the data points
ax.scatter(x1, x2, x3, color='red', marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.set_title('Regression Plane')

plt.show()