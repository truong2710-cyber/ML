import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from display_network import *

mndata=MNIST('D:/MNIST')
mndata.load_testing()
X=mndata.test_images
X0 = np.asarray(X)[:1000,:]/256.0
X=X0

K=10
kmeans=KMeans(n_clusters=K).fit(X)
pred_label=kmeans.predict(X)
A = display_network(kmeans.cluster_centers_.T, K, 1)

f1 = plt.imshow(A, interpolation='nearest', cmap = "jet")
f1.axes.get_xaxis().set_visible(False)
f1.axes.get_yaxis().set_visible(False)
plt.show()


