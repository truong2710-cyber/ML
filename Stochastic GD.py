#test linear regression with SGD

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X=np.random.rand(1000,1)
#add noise
y=4+3*X+0.2*np.random.randn(1000,1)

#Buiding Xbar
one=np.ones((1000,1))
Xbar=np.concatenate((one,X),axis=1)

A=np.dot(Xbar.T,Xbar)
b=np.dot(Xbar.T,y)
w_lr=np.dot(np.linalg.pinv(A),b)
print("Solution found by formula:            w = ",w_lr.T)

def grad(w):
    N=X.shape[0]
    return 1/N*np.dot(Xbar.T,(np.dot(Xbar,w)-y))

def sgrad(w,i,rd_id):
    true_i=rd_id[i]
    xi=Xbar[true_i]
    yi=y[true_i]
    a=np.dot(xi,w)-yi
    return (xi*a).reshape(2,1)

def cost(w):
    N=X.shape[0]
    return (0.5/N)*(np.linalg.norm(y-np.dot(Xbar,w))**2)

def SGD(w_init,sgrad,eta,epochs):
    w=[w_init]
    iter_check_w=10
    w_last_check = w_init
    N=X.shape[0]
    count=0
    for epoch in range(epochs):
        rd_id=np.random.permutation(N)  #shuffle the dataset
        for i in range(N):
            count+=1
            g=sgrad(w[-1],i,rd_id)
            w_new=w[-1]-eta*g
            w.append(w_new)
            if count%iter_check_w == 0:
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    return w
                w_last_check = w_this_check
    return w

w_init=np.array([[2],[1]])
epochs=10
print("Solution found by SGD after %i epochs: w = "%epochs,SGD(w_init,sgrad,0.1,epochs)[-1].T)

#display result
w0=w_lr[0][0]
w1=w_lr[1][0]
x0=np.linspace(0,1,2)
y0=w0+w1*x0

plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'r', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()