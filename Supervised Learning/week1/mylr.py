#This Program Is My First Attempt To Make A Cost Fn All By Myself

import numpy as np
import matplotlib.pyplot as plt
from random import randint

x_train=np.array([2,3,4,5,8,6])
y_train=np.array([100,350,370,420,870,750])


l=[]
l2=[]
m=x_train.shape[0]  #gives the no of datasets
def costf(m,w,b):
    cost=[]
    for i in range(m):  #for each value of x calculates the cost and add to cost[]
        cost.append((1/(2*m))*(((w*x_train[i]+b)-y_train[i])**2))
    l.append(sum(cost)) #sum of cost[] gives net cost
    l2.append([w,b])    #to preserve the value of w and b for every append
def model():
    for w in range(1000):
        for b in range(-1000,1000):
            costf(m,w,b)    #call the function for varying value of w & b
    w=[l2[l.index(min(l))]][0][0]   # min value of l then index of that val corresponding
    b=[l2[l.index(min(l))]][0][1]   #,to l2 to obtain preserved val of w and b
    print(w,b)
    fb=np.zeros(m)  #creates an empty tensor of shape(2,0,0)
    for i in range(m):  #generating the final vals of y^ for each value of x
        fb[i]=w*x_train[i]+b
    return fb

fb=model()
plt.plot(x_train,fb,c='blue',label='Our Prediction')
plt.scatter(x_train,y_train,c='red',marker='x',label='Actual inputs')
plt.legend()
plt.show()
