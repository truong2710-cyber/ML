import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(1)
means=[[2,2],[8,3],[3,6]]
cov=[[1,0],[0,1]]
N=500
X0=np.random.multivariate_normal(means[0],cov,N)
X1=np.random.multivariate_normal(means[1],cov,N)
X2=np.random.multivariate_normal(means[2],cov,N)
X=np.concatenate((X0,X1,X2),axis=0)
K=3
original_label=np.asarray([0]*N+[1]*N+[2]*N).T

def kmeans_display(X,label,centers):
    K=np.amax(label)+1
    X0=X[label==0,:]
    X1=X[label==1,:]
    X2=X[label==2,:]
    plt.plot(X0[:,0],X0[:,1],'b^',markersize = 4,alpha=.5)
    plt.plot(X1[:,0],X1[:,1],'go',markersize = 4,alpha=.5)
    plt.plot(X2[:,0],X2[:,1],'rs',markersize = 4,alpha=.5)
    for c in centers:
        plt.plot([c[0]],[c[1]],'yo',markersize=12) 
    plt.axis('equal')
    plt.plot()
    plt.show()



def kmeans_init_center(X,k):
    return X[np.random.choice(X.shape[0],k,replace=False)]

def kmeans_assign_labels(X,centers):
    D=cdist(X,centers)
    return np.argmin(D,axis=1)

def kmeans_update_centers(X,labels,K):
    centers=np.zeros((K,X.shape[1]))
    for k in range(K):
        Xk=X[labels==k,:]
        centers[k,:]=np.mean(Xk,axis=0)
    return centers

def has_converged(centers, new_centers):
    return(set([tuple(a) for a in centers])==set([tuple(a) for a in new_centers]))

def kmeans(X,K):
    centers=kmeans_init_center(X,K)
    while True:
        labels=kmeans_assign_labels(X,centers)
        new_centers=kmeans_update_centers(X,labels,K)
        if has_converged(centers,new_centers):
            break
        centers=new_centers
    return (centers,labels)

centers,labels=kmeans(X,K)
print(centers)

kmeans_display(X,labels,centers)