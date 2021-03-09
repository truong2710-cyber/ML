import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target
print("Number of classes: {}".format(np.unique(iris_y)))
X_train, X_test, y_train, y_test=train_test_split(iris_X,iris_y,train_size=100)
print("Training_size: {}".format(len(X_train)))
print("Test_size    : {}".format(len(X_test)))

clf=neighbors.KNeighborsClassifier(n_neighbors=10,p=2, weights='distance')
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Result for 20 test data points:")
print("Predicted labels:",y_pred[20:40])
print("Ground truth    :",y_test[20:40])
print("Accuracy of 10NN (weights = distance): %.2f %%" %(100*accuracy_score(y_test,y_pred)))
