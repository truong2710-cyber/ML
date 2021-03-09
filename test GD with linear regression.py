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
print("Solution found by formula: w = ",w_lr.T)

#display result
w0=w_lr[0][0]
w1=w_lr[1][0]
x0=np.linspace(0,1,2)
y0=w0+w1*x0

plt.plot(X.T, y.T, 'b.')     # data 
plt.plot(x0, y0, 'r', linewidth = 2)   # the fitting line
plt.axis([0, 1, 0, 10])
plt.show()

def grad(w):
    N=X.shape[0]
    return 1/N*np.dot(Xbar.T,(np.dot(Xbar,w)-y))

def cost(w):
    N=X.shape[0]
    return (0.5/N)*(np.linalg.norm(y-np.dot(Xbar,w))**2)

def numberical_grad(w,cost):
    eps = 1e-4
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_p = w.copy()
        w_n = w.copy()
        w_p[i] += eps 
        w_n[i] -= eps
        g[i] = (cost(w_p) - cost(w_n))/(2*eps)
    return g 

def check_grad(w,cost,grad):
    grad1=grad(w)
    grad2=numberical_grad(w,cost)
    return True if np.linalg.norm(grad1-grad2)<1e-6 else False

def GD(w_init,grad,eta):
    w=[w_init]
    for it in range(100):
        w_new=w[-1]-eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))<1e-6:
            break
        w.append(w_new)
    return (w[-1],it)

w_init=np.array([[2],[1]])
(w1,it1)=GD(w_init,grad,1)
print("Solution found by GD:      w = ",w1.T)
print("Get the solution after %i iterations(epochs)."%(it1+1))