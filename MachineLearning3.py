import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
#This is a kmeans clustering model that I am working on

X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1,0.6],
              [9,11]])

plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

clf_kme = KMeans(n_clusters=2)
clf_kme.fit(X)

centroids = clf_kme.cluster_centers_
labels = clf_kme.labels_

colors = ["g.", "r.", "c.", "y.", "b."]