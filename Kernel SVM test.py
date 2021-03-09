import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from cvxopt import matrix, solvers
np.random.seed(12)
means=[[2,2],[3,2]]
cov=[[0.5,0],[0,0.5]]
N=20
X0=np.random.multivariate_normal(means[0],cov,N)
X1=np.random.multivariate_normal(means[1],cov,N)
X=np.concatenate((X0.T,X1.T),axis=1)
label=np.array([1.]*N+[-1.]*N)

clf=SVC(C=100,kernel="poly")
clf.fit(X.T,label)
x=np.linspace(-2.0,5.0,100)
y=np.linspace(-2.0,5.0,100)
XX,YY=np.meshgrid(x,y)
Z=clf.decision_function(np.c_[XX.ravel(),YY.ravel()])
Z=Z.reshape(XX.shape)
CS = plt.contourf(XX, YY, np.sign(Z), 200, cmap='jet', alpha = .2)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])
for x in X0:
    plt.plot([x[0]],[x[1]],"r^")
for x in X1:
    plt.plot([x[0]],[x[1]],"bo")
plt.xlim([-2,5])
plt.ylim([-2,5])
plt.show()