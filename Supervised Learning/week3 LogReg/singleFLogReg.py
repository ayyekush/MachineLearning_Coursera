# My attempt on logistic regression(classification)

import matplotlib.pyplot as plt
import numpy as np
from math import exp

x=np.array([-2,-1,3,4,5,6])
y=np.array([0,0,0,1,1,1])
m=len(x)
def sig(w,b):
    fx1=[]
    for k in range(6):
        fx1.append(1/(1+exp(-(w*x[k]+b))))
    fx=np.array(fx1)
    return fx
print(sig(1,0))
def gd(x,y,epoch,alpha):
    w=1
    b=1
    for i in range(epoch):
        fxd=sig(w,b)
        #-1/fx x*(fxd**2)
        # dw=1/m*(np.sum(((fxd-y)**2)*(x*(fxd**2))))
        dw=1/m*(np.sum((fxd-y)*x))
        db=1/m*(np.sum((fxd-y)))
        # db=1/m*(np.sum(((fxd-y)**2)*((fxd**2))))
        w-=alpha*dw
        b-=alpha*db
    return w,b
w,b=gd(x,y,10000,0.01)
print(w,b)
plt.plot(x,sig(w,b),label='Sigmoid Function')
plt.plot([1,6],[0.5,0.5],label='Decision Boundary')
plt.scatter(x,y)
plt.ylabel('Z axis')
plt.xlabel('X axis')
plt.legend()
plt.grid(True)
plt.show()