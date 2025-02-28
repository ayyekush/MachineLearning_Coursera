# This is a heavily smodified version of multivar lr provided in course

import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
# b_init = 785.1811367994083
# w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
#/ m = X_train.shape[0]#no of rows of training data
#/ n = X_train[0].shape[0]#diff features for each data (x1,x2,x3,x4)

def predict(w,b):
    y1,y2,y3=np.dot(X_train[0],w)+b,  np.dot(X_train[1],w)+b,  np.dot(X_train[2],w)+b
    return y1,y2,y3

def compute_cost(X,y,w, b):
    m,n=X.shape
    cost = 0.0
    for i in range(m):#for each row                         
        f_wb= np.dot(X[i], w) + b
        cost += (f_wb - y[i])**2 #sum of differences squared
    cost = cost / (2 * m)
    return cost

# # Compute and display cost using our pre-chosen optimal parameters. 
# cost = compute_cost(X_train, y_train, w_init, b_init)
# print(f'Cost at optimal w : {cost}')

def compute_gradient(X,y, w, b): 
    m,n=X.shape
    dj_dw = np.zeros(n) #zero tensor for all wes (w1,w2,w3,w4)
    dj_db = 0.0
    for i in range(m): # for every row in training set                        
        error = np.dot(X[i], w) + b - y[i]
        ## 1up training set i:[x1,x2,x3,x4] weighted and biased which 
            # predict y[i] then subtracted from real y[i]
        dj_db += error # making value of b cover up for the error  
        #since dj/db = 1/m sum(error(i)) #remember error has hidden - y                    
        dj_dw +=  error * X[i] 
        #since dj/dw1 = 1/m sum((error(i))*x1(i))
    dj_dw = dj_dw / m
    dj_db = dj_db / m  
    return dj_db, dj_dw
#Compute and display gradient 
# tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
# print(f'dj_db at initial w,b: {tmp_dj_db}')
# print(f'dj_dw at initial w,b(very close to zero,as they should be): {tmp_dj_dw}')

#starter_w = np.zeros(n) #for each feature(x)
# starter_b = 0.0
#you should preserve your intial values as it is not always zero
epoch = 1000
alpha = 5.0e-7

def gradient_descent(X,y,epoch,alpha):
    m,n=X.shape
    J_history = []
    w = np.zeros(n) #avoid modifying initial w
    b = 0.0
    for i in range(epoch):
        # Calculate the gradient and update the parameters
        dj_db,dj_dw = compute_gradient(X,y, w, b) 
        # Update Parameters using w, b, alpha and gradient
        w -= alpha * dj_dw #gradient decent for each w
        b -= alpha * dj_db
        J_history.append(compute_cost(X,y, w, b))
        if i % 99 == 0:
            print(f"Iteration {i:3d}: Cost {J_history[-1]:7.2f}")
            # 3d just add extra spaces to make any integer 3 'digit' long
            # 7.2f 7 is minimum width just like 1u and 2 places after decimal
    return w, b, J_history

if __name__ == '__main__':
    m,n=X_train.shape
    w_final, b_final, J_hist = gradient_descent(X_train,y_train,epoch,alpha)
    print(f"\nb,w found by gradient descent: {b_final:0.2f} , [{w_final[0]:0.4f},{w_final[1]:0.4f},{w_final[2]:0.4f},{w_final[3]:0.4f}]")
    print("final cost: ",J_hist[-1])
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    #subplot creates a matrix of individual sub-figures
        #,1*2=rows*colums of matrix,figsize is size ration of whole figure
    fig, (subFig1, subFig2) = plt.subplots(1, 2, figsize=(19, 4))
    #tuple returned by subplots===(wholeFigure,subFig1,subFig2)
    subFig1.plot(J_hist)# when x-axis isnt given it assumes the indeces of jhist to be xaxis
    subFig2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    subFig1.set_title("Cost vs. iteration");  subFig2.set_title("Cost vs. iteration (lower half maximised)")
    subFig1.set_ylabel('Cost')             ;  subFig2.set_ylabel('Cost') 
    subFig1.set_xlabel('iteration step')   ;  subFig2.set_xlabel('iteration step') 
    plt.show()

# b,w found by gradient descent: -0.00 , [0.2040,0.0037,-0.0112,-0.0659]