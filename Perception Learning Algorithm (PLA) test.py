import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(0)

means=[[2,2],[4,2]]
cov=[[.3,.2],[.2,.3]]
N=10
X0=np.random.multivariate_normal(means[0],cov,N).T
X1=np.random.multivariate_normal(means[1],cov,N).T
X=np.concatenate((X0,X1),axis=1)
one=np.ones((1,X.shape[1]))
X=np.concatenate((one,X),axis=0)
y=np.concatenate((np.ones((1,N)),-1*np.ones((1,N))),axis=1)

def h(w,x):
    return np.sign(np.dot(w.T,x))

def has_converged(X,y,w):
    return np.array_equal(h(w,X),y)

def PLA(X,y,w_init):
    w=[w_init]
    N=X.shape[1]
    d=X.shape[0]
    M=[] #misclassified points
    while True:
        rd_id=np.random.permutation(N)
        for i in range(N):
            if np.dot(w[-1].T,X[:,rd_id[i]].reshape(d,1))*y[0,rd_id[i]]<0:
                M.append(rd_id[i])
                w_new=w[-1]+y[0,rd_id[i]]*X[:,rd_id[i]].reshape(d,1)
                w.append(w_new)
        if has_converged(X,y,w[-1]):
            break
    return (w,M)

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = PLA(X, y, w_init)
w_b=w[-1]
print(w_b)
for x in X0.T:
    plt.plot([x[0]],[x[1]],"r^")
for x in X1.T:
    plt.plot([x[0]],[x[1]],"bo")
xl=np.array([0,5])
yl=-w_b[0][0]/w_b[2][0]-w_b[1][0]*xl/w_b[2][0]
plt.plot(xl,yl)

plt.show()
