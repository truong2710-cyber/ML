import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import accuracy_score

# Create 3 classes which are not linearly separable
N=100 # number of points per class
d0=2  # input dimension
C=3   # number of class
X=np.zeros((d0,N*C)) # data matrix (each column = 1 point)
y=np.zeros(N*C)      # data labels 
for j in range(C):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[:,ix] = np.concatenate((r*np.sin(t).reshape(1,N), r*np.cos(t).reshape(1,N)),axis=0)
    y[ix] = j
# lets visualize the data:
# plt.scatter(X[:N, 0], X[:N, 1], c=y[:N], s=40, cmap=plt.cm.Spectral)gg

plt.plot(X[0, :N], X[1, :N], 'bs', markersize = 7)
plt.plot(X[0, N:2*N], X[1, N:2*N], 'ro', markersize = 7)
plt.plot(X[0, 2*N:], X[1, 2*N:], 'g^', markersize = 7)
# plt.axis('off')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
cur_axes = plt.gca()
cur_axes.axes.get_xaxis().set_ticks([])
cur_axes.axes.get_yaxis().set_ticks([])

plt.savefig('EX.png', bbox_inches='tight', dpi = 600)


def softmax(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each column of Z is a set of score.    
    """
    e_Z=np.exp(Z-np.max(Z,axis=0,keepdims=True))
    A=e_Z/e_Z.sum(axis=0)
    return A

def convert_labels(y,C=C):
    """
    convert 1d label to a matrix label: each column of this 
    matrix coresponding to 1 element in y. In i-th column of Y, 
    only one non-zeros element located in the y[i]-th position, 
    and = 1 ex: y = [0, 2, 1, 0], and 3 classes then return

            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    """
    Y=sparse.coo_matrix((np.ones_like(y),(y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

def cost(Y,Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]

d0=2
d1=h=100 #size of hidden layer
d2=C=3

#initialize parameter randomly
W1=0.01*np.random.randn(d0,d1)
b1=np.zeros((d1,1))
W2=0.01*np.random.randn(d1,d2)
b2=np.zeros((d2,1))

Y=convert_labels(y)
N=X.shape[1]
eta=1
for i in range(10000):
    # Feedforward
    Z1=W1.T.dot(X)+b1
    A1=np.maximum(Z1,0)
    Z2=W2.T.dot(A1)+b2
    A2=softmax(Z2)
    Yhat=A2
    if i %1000 == 0:
        # compute the loss: average cross-entropy loss
        loss = cost(Y, Yhat)
        print("iter %d, loss: %f" %(i, loss))
    # backpropagation
    E2=1/N*(Yhat-Y)
    dW2=A1.dot(E2.T)
    db2=np.sum(E2,axis=1,keepdims=True)
    E1=(W2.dot(E2))
    E1[Z1<0]=0
    dW1=X.dot(E1.T)
    db1=np.sum(E1,axis=1,keepdims=True)
    # Gradient Descent
    W1-=eta*dW1
    b1-=eta*db1
    W2-=eta*dW2
    b2-=eta*db2
# caculate predicted class and accuracy 
Z1=np.dot(W1.T,X)+b1
A1=np.maximum(Z1,0)
Z2=W2.T.dot(A1)+b2
A2=softmax(Z2)
Yhat=A2
predicted_class=np.argmax(Yhat,axis=0)

print("Accuracy score is %.2f %%"%(100*accuracy_score(predicted_class,y)))

xm=np.arange(-1.5,1.5,0.03)
ym=np.arange(-1.5,1.5,0.03)
xx,yy=np.meshgrid(xm,ym)
xx1=xx.ravel().reshape(1,100**2)
yy1=yy.ravel().reshape(1,100**2)
X0=np.vstack((xx1,yy1))
Z1 = np.dot(W1.T, X0) + b1 
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
# predicted class 
Z = np.argmax(Z2, axis=0)
Z=Z.reshape(xx.shape)
CS=plt.contourf(xx,yy,Z,200,cmap='jet',alpha=.1)

plt.show()






