import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(0)

# randomly generate data
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# each column is a datapoint
X = np.concatenate((X0, X1, X2), axis = 0).T 
# extended data
X = np.concatenate((np.ones((1, 3*N)), X), axis = 0)
C = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N)

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

Y=convert_labels(original_label,C)

def cost(X,Y,W):
    A=softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A))

def grad(X,Y,W):
    A=softmax(W.T.dot(X))
    E=A-Y
    return X.dot(E.T)

def softmax(Z):
    """
    Compute softmax values for each sets of scores in Z.
    each column of Z is a set of score.    
    """
    e_Z=np.exp(Z)
    A=e_Z/e_Z.sum(axis=0)
    return A

def softmax_regression(X,y,W_init,eta,tol=1e-6,max_count=10000):
    Y=convert_labels(y)
    W=[W_init]
    N=X.shape[1]
    d=X.shape[0]
    count=0
    W_last_check=W[-1]
    check_W_after=20
    while count<max_count:
        mix_id=np.random.permutation(N)
        for i in mix_id:
            count+=1
            xi=X[:,i].reshape(d,1)
            A=softmax(W[-1].T.dot(X))
            ai=A[:,i].reshape(C,1)
            yi=Y[:,i].reshape(C,1)
            ei=yi-ai
            W_new=W[-1]+eta*(xi.dot(ei.T))
            W.append(W_new)
            W_this_check=W_new
            if count%check_W_after==0:
                if np.linalg.norm(W_this_check-W_last_check)<tol:
                    break
                W_last_check=W_this_check
    return W

eta=.05
W_init=np.random.randn(X.shape[0], C)
W=softmax_regression(X,original_label,W_init,eta)
print(W[-1])

def predict(W,X):
    """
    predict output of each columns of X
    Class of each x_i is determined by location of max probability
    Note that class are indexed by [0, 1, 2, ...., C-1]
    """
    A=softmax(W.T.dot(X))
    return np.argmax(A,axis=0)

print(predict(W[-1],X))
print(original_label)
print("Accuracy score: %.2f %%"%(100*(accuracy_score(predict(W[-1],X),original_label))))


# display result
Wmax=W[-1]
w0=Wmax[:,0]
w1=Wmax[:,1]
w2=Wmax[:,2]

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return [x,y]
    else:
        return False
def boundary(w,x):
    return -w[0]/w[2]-w[1]*x/w[2]

def display():
    K=np.amax(original_label)+1
    X0=X[:,original_label==0]
    X1=X[:,original_label==1]
    X2=X[:,original_label==2]
    plt.plot(X0[1,:],X0[2,:],'b^',markersize = 4,alpha=.5)
    plt.plot(X1[1,:],X1[2,:],'go',markersize = 4,alpha=.5)
    plt.plot(X2[1,:],X2[2,:],'rs',markersize = 4,alpha=.5)
    w01=w0-w1
    x=np.array([-5,15])
    y1=-w01[0]/w01[2]-w01[1]*x/w01[2]
    w12=w1-w2
    y2=-w12[0]/w12[2]-w12[1]*x/w12[2]
    w20=w2-w0
    L1=line([x[0],y1[0]],[x[1],y1[1]])
    L2=line([x[0],y2[0]],[x[1],y2[1]])
    x_intersect=intersection(L1,L2)[0]
    plt.plot([x_intersect,5],boundary(w01,np.array([x_intersect,5])))
    plt.plot([x_intersect,-15],boundary(w20,np.array([x_intersect,-15])))
    plt.plot([x_intersect,15],boundary(w12,np.array([x_intersect,15])))
    plt.axis('equal')
    plt.plot()
    plt.show()

display()


