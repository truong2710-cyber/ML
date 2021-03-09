import numpy as np
#naive way to calculate loss and gradient

def svm_loss_naive(W,X,y,reg): #reg is regularization factor
    d,C=W.shape
    d,N=X.shape
    loss=0
    dW=np.zeros(W.shape)
    for n in range(N):
        xn=X[:,n]
        z=W.T.dot(xn)
        for j in range(C):
            if j!=y[n]:
                margin=1-z[y[n]]+z[j]
                if margin>0:
                    dW[:,y[n]]-=xn
                    dW[:,j]+=xn
                    loss+=margin
    loss=loss/N+reg/2*np.linalg.norm(W)
    grad=dW/N+reg*W
    return loss,grad

N,C,d=10,3,5
reg=.1
W = np.random.randn(d, C)
X = np.random.randn(d, N)
y = np.random.randint(C, size = N)
print('loss without regularization:', svm_loss_naive(W, X, y, 0)[0])
print('loss with regularization (naive way):',svm_loss_naive(W, X, y, .1)[0])

# a more effective way to calculate loss and grad
def svm_loss_vectorized(W,X,y,reg):
    d,C=W.shape
    d,N=X.shape
    loss=0
    dW=np.zeros(W.shape)
    Z=W.T.dot(X)
    correct_class_score=np.choose(y,Z).reshape(1,N) # value of scores of the correct classes
    margins = np.maximum(0, Z - correct_class_score + 1) # loss matrix
    margins[y,np.arange(N)]=0 # set loss at correct classes = 0 
    loss=np.sum(margins)
    loss=loss/N+reg/2*np.linalg.norm(W)
    F = (margins > 0).astype(int) 
    F[y, np.arange(F.shape[1])] = np.sum(-F, axis = 0)
    dW = X.dot(F.T)/N + reg*W
    return loss, dW
print('loss with regularization (effective way):',svm_loss_vectorized(W, X, y, .1)[0])

def multiclass_svm_minibatchGD(X,y,Winit,reg,eta=.1,batch_size=100,num_iter=1000,print_after=100):
    W=Winit
    for i in range(num_iter):
        idx = np.random.choice(X.shape[1], batch_size)
        X_batch = X[:, idx]
        y_batch = y[idx]
        W-=eta*svm_loss_vectorized(W,X_batch,y_batch,reg)[1]
        if i%print_after==0:
            print("Loss at {}/{} iter: {}".format(i,num_iter,svm_loss_vectorized(W,X_batch,y_batch,reg)[0]))
    
    return W

N, C, d = 50, 4, 10
reg = .1 
Winit = np.random.randn(d, C)
X = np.random.randn(d, N)
y = np.random.randint(C, size = N)
print("W = \n",multiclass_svm_minibatchGD(X,y,Winit,reg))