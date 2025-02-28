import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-darkgrid")

example=np.array([[[9,2,0],[3,4,4]],[[3,4,9],[4,5,4]],[[3,4,9],[4,5,4]]])
# print(f"example indeces: {example[1][1]}");
# print(f"example.shape: {example.shape}"); #1//shape = (no of rows[index1],no of ele in each row[index0],no of ele in each ele of each row[index2])
# print(f"shape indices in #1: {example.shape[1]}"); 

# Question: plot a graph (1k sqft,$300k) ,, (2k sqft,$500k)
x_train=np.array([1,2])#this tensor has two rows even if it doesnt look like
y_train=np.array([300,500])

def model():
    w=250
    b=20
    f_wb = np.zeros(x_train.shape[0])#initialising an empty array
    #shape[0] is just selecting the number of rows from shape tuple
    for i in range(x_train.shape[0]):
        f_wb[i]=w*x_train[i]+b
    return f_wb

plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.scatter(x_train, y_train, marker='x', c='red',label='Actual Values')
plt.plot(x_train, model(),marker='o', c='blue',label='Our Prediction')

plt.legend() # nigga shows the teams switchtfo
plt.show()