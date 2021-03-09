import numpy as np
from sklearn import linear_model            #for logistic regression
from sklearn.metrics import accuracy_score  #for evaluation
from scipy import misc                      #for loading image
from scipy.io import loadmat

def feature_extraction(X):
    return (X - x_mean)/x_var     

annots = loadmat('D:\\Python\\ML\\AR\\randomfaces4ar.mat')
np.random.seed(0)

train_ids = np.arange(1, 26)
test_ids = np.arange(26, 50)
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21)))
X=annots['featureMat']
y=annots['labelMat']
z=annots['filenameMat']
X_train_full=np.concatenate((X[:,0:650],X[:,1300:1950]),axis=1)
X_test_full=np.concatenate((X[:,650:1300],X[:,1950:2600]),axis=1)
x_mean = X_train_full.mean(axis = 0)
x_var  = X_train_full.var(axis = 0)
X_train = feature_extraction(X_train_full)
X_test  = feature_extraction(X_test_full)
#X_train=np.concatenate((np.ones((1,1300)),X_train),axis=0)
y_train=np.array([1]*650+[0]*650)
y_test =np.array([1]*650+[0]*650)
logreg=linear_model.LogisticRegression(C=1e5,max_iter=900)
logreg.fit(X_train.T,y_train.T)
y_pred = logreg.predict(X_test.T)
print ("Accuracy: %.2f %%" %(100*accuracy_score(y_test.T, y_pred)))



