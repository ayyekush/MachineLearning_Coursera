# ITP single variable gradient decent
import numpy as np
import matplotlib.pyplot as plt


x_train = np.array([2, 3, 4, 5, 8, 6]);
y_train = np.array([100, 350, 370, 420, 870, 750]);
learning_rate = 0.001;
epochs = 100000;
w = 0;b = 0;
m=x_train.shape[0];
print(m)
# Perform gradient descent
for i in range(epochs):
    # derivative of our cost function (J=1/2m.sum((w*x+b)-y)**2)
    dw = (1/m)*(np.sum(((w*x_train+b)-y_train)*x_train))
    db = (1/m)*(np.sum((w*x_train+b)*y_train))
    w -= learning_rate * dw
    b -= learning_rate * db
    
print("Values of w and b after gradient descent:")
print("w =", w)
print("b =", b)
# Plotting the data points and the regression line
plt.scatter(x_train, y_train, color='blue', label='Data Points')
plt.plot(x_train, (np.dot(w,x_train))+b, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()