import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import accuracy_score
np.random.seed(0)

# Create 3 classes which are not linearly separable
N=20  # number of points per class
d0=2  # input dimension
C=3   # number of class
means = [[-1, -1], [1, -1], [0, 1]]
cov = [[1, 0], [0, 1]]
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
X = X.T
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N)
def kmeans_display(X,label):
    K=np.amax(label)+1
    X0=X[:,label==0]
    X1=X[:,label==1]
    X2=X[:,label==2]
    plt.plot(X0[0,:],X0[1,:],'b^',alpha=.8,markersize=6)
    plt.plot(X1[0,:],X1[1,:],'go',alpha=.8,markersize=6)
    plt.plot(X2[0,:],X2[1,:],'rs',alpha=.8,markersize=6)
    plt.show()

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

def cost(Y,Yhat,W1,W2,lamda):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]+lamda*(np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2)

def mynet(lamda):
    d0=2
    d1=h=100 #size of hidden layer
    d2=C=3

    #initialize parameter randomly
    W1=0.01*np.random.randn(d0,d1)
    b1=np.zeros((d1,1))
    W2=0.01*np.random.randn(d1,d2)
    b2=np.zeros((d2,1))

    Y=convert_labels(original_label)
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
            loss = cost(Y, Yhat,W1,W2,lamda)
            print("iter %d, loss: %f" %(i, loss))
        # backpropagation
        E2=1/N*(Yhat-Y)
        dW2=A1.dot(E2.T)+lamda*W2
        db2=np.sum(E2,axis=1,keepdims=True)
        E1=(W2.dot(E2))
        E1[Z1<0]=0
        dW1=X.dot(E1.T)+lamda*W1
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

    print("Accuracy score is %.2f %%"%(100*accuracy_score(predicted_class,original_label)))

    xm=np.arange(-4,4,0.08)
    ym=np.arange(-4,4,0.08)
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
    kmeans_display(X,original_label)
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    

mynet(0.01)






