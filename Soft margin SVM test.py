import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from cvxopt import matrix, solvers
np.random.seed(1)
means=[[2,2],[3,2]]
cov=[[0.3,0.2],[0.2,0.3]]
N=20
X0=np.random.multivariate_normal(means[0],cov,N)
X1=np.random.multivariate_normal(means[1],cov,N)
X=np.concatenate((X0.T,X1.T),axis=1)
y=np.array([1.]*N+[-1.]*N)

C=100

# using sklearn tool
clf=SVC(kernel='linear',C=C)
clf.fit(X.T,y)
w=clf.coef_
b=clf.intercept_[0]
print(w,b)

# using Lagrange duality and cvxopt tool
V=np.concatenate((X0.T,-X1.T),axis=1)
P=matrix(V.T.dot(V))
q=matrix(-np.ones((2*N,1)))
G=matrix(np.concatenate((-np.eye(2*N),np.eye(2*N)),axis=0))
h=matrix(np.array([.0]*2*N+[float(C)]*2*N).T)
A=matrix(y.reshape((1,2*N)))
b=matrix(np.zeros((1,1)))
solvers.options['show_progress']=False
sol=solvers.qp(P,q,G,h,A,b)
lamda=np.array(sol['x'])

S=np.where(lamda>1e-5)[0] # support vectors
S1=np.where(lamda<.999*C)[0]
M=[x for x in S if x in S1] # vectors on margins
lamdaS=lamda[S]
VS=V[:,S]
yM=y[M]
XM=X[:,M]
w=VS.dot(lamdaS)
b=np.mean(yM-w.T.dot(XM))
print(w.T,b)
for x in X0:
    plt.plot([x[0]],[x[1]],"bo")
for x in X1:
    plt.plot([x[0]],[x[1]],"r^")
x1 = np.arange(-10, 10, 0.1)
y1 = -w[0, 0]/w[1, 0]*x1 - b/w[1, 0]
y2 = -w[0, 0]/w[1, 0]*x1 - (b-1)/w[1, 0]
y3 = -w[0, 0]/w[1, 0]*x1 - (b+1)/w[1, 0]
plt.plot(x1, y1, 'k', linewidth = 3)
plt.plot(x1, y2, 'k')
plt.plot(x1, y3, 'k')
y4 = 100*x1
plt.plot(x1, y1, 'k')
plt.fill_between(x1, y1, color='red', alpha=.3)
plt.fill_between(x1, y1, y4, color = 'blue', alpha = .3)
plt.xlim([0.5,4])
plt.ylim([0.5,4])


# using GD
X0_bar=np.vstack((X0.T,np.ones((1,N))))
X1_bar=np.vstack((X1.T,np.ones((1,N))))
Z = np.hstack((X0_bar, - X1_bar))
lamb=1./C

def cost(w):
    u=w.T.dot(Z)
    return np.sum(np.maximum(0,1-u))+lamb/2 * (np.linalg.norm(w)-w[-1]*w[-1])

def grad(w):
    u=w.T.dot(Z)
    H=np.where(u<1)[1]
    ZH=Z[:,H]
    g=-np.sum(ZH,axis=1,keepdims=True)+lamb*w
    g[-1]-=lamb*w[-1]
    return g

def GD(w0,eta):
    w=w0
    for i in range(100000):
        w=w-eta*grad(w)
        if np.linalg.norm(grad(w))<1e-5:
            break
    return w

w0=np.array([-3,4,5]).reshape(3,1)
eta=.001
w_bar=GD(w0,eta)
print(w_bar[:2,0],end=" ")
print(w_bar[-1,0])
plt.show()


