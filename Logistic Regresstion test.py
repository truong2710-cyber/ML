import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

X=np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y=np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
X=np.concatenate((np.ones((1,X.shape[1])),X),axis=0)
def sigmoid(s):
    return 1/(1+np.exp(-s))

def logistic_sigmoid_regression(X,y,w_init,eta):
    w=[w_init]
    it=0
    N=X.shape[1]
    d=X.shape[0]
    count=0
    check_w_after=20
    w_last_check=w_init
    while count<10000:
        mix_id=np.random.permutation(N)
        for i in mix_id:
            xi=X[:,i].reshape(d,1)
            yi=y[i]
            zi=sigmoid(np.dot(w[-1].T,xi))
            w_new=w[-1]+eta*(yi-zi)*xi
            w.append(w_new)
            count+=1
            if count % check_w_after==0:
                w_this_check=w_new
                if np.linalg.norm(w_this_check-w_last_check)<1e-6:
                    break
                w_last_check=w_this_check
    return w

eta = .05 
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])
print(sigmoid(np.dot(w[-1].T,X)))

#Show graph illustration
N=X.shape[1]
for i in range(N):
    if y[i]==0:
        plt.plot([X[1][i]],[y[i]],"ro")
    else:
        plt.plot([X[1][i]],[y[i]],"bs")
X_test=np.array([np.linspace(0,6,1000)])
X_test=np.concatenate((np.ones((1,X_test.shape[1])),X_test),axis=0)
y_test=sigmoid(np.dot(w[-1].T,X_test))
plt.plot(X_test[1],y_test[0])
threshold = -w[-1][0][0]/w[-1][1][0]
plt.plot(threshold, .5, 'y^', markersize = 8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()

            
