import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist
from cvxopt import matrix,solvers
from sklearn.svm import SVC
np.random.seed(1)

means=[[2,2],[5,5]]
cov=[[1,0],[0,1]]
N=20
X0=np.random.multivariate_normal(means[0],cov,N)
X1=np.random.multivariate_normal(means[1],cov,N)
X=np.concatenate((X0.T,X1.T),axis=1)
y=np.array([[1.]*N+[-1.]*N])

#build K
V=np.concatenate((X0.T,-X1.T),axis=1)
K=matrix(V.T.dot(V))
p=matrix(-np.ones((2*N,1)))

#build A,b,G,h
A=matrix(y)
b=matrix(np.zeros((1,1)))
G=matrix(-np.eye(2*N))
h=matrix(np.zeros((2*N,1)))
solvers.options['show_progress']=False
sol=solvers.qp(K,p,G,h,A,b)
l=np.array(sol['x'])
epsilon=1e-6
S=np.where(l>epsilon)[0]
VS=V[:,S]
XS=X[:,S]
yS=y[:,S]
lS=l[S]
w=V.dot(l)
b=np.mean(yS-w.T.dot(XS))
print("w = ",w.T)
print("b = ",b)


fig, ax = plt.subplots()
x1 = np.arange(-10, 10, 0.1)
y1 = -w[0, 0]/w[1, 0]*x1 - b/w[1, 0]
y2 = -w[0, 0]/w[1, 0]*x1 - (b-1)/w[1, 0]
y3 = -w[0, 0]/w[1, 0]*x1 - (b+1)/w[1, 0]
plt.plot(x1, y1, 'k', linewidth = 3)
plt.plot(x1, y2, 'k')
plt.plot(x1, y3, 'k')
y4 = 100*x1
plt.plot(x1, y1, 'k')
plt.fill_between(x1, y1, color='red', alpha=.1)
plt.fill_between(x1, y1, y4, color = 'blue', alpha = .1)
for m in S:
    circle = plt.Circle((X[0, m], X[1, m] ), 0.1, color='k', fill = False)
    ax.add_artist(circle)
for x in X0:
    plt.plot([x[0]],[x[1]],"rs")
for x in X1:
    plt.plot([x[0]],[x[1]],"bo")
xl=np.array([0,7])
yl=-b/w[1][0]-w[0][0]*xl/w[1][0]
plt.plot(xl,yl)
plt.xlim([0.5,7])
plt.ylim([0.5,7])
plt.show()
