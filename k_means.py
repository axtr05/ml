import numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import KMeans

x1=np.array([[1,1],[1.5,2],[3,4],[5,7],[3.5,5],[4.5,5],[3.5,4]])
x2=np.random.rand(20,2)*10

k1=KMeans(n_clusters=2,random_state=0).fit(x1)
k2=KMeans(n_clusters=3,random_state=1).fit(x2)

plt.subplot(1,2,1)
plt.scatter(x1[:,0],x1[:,1],c=k1.labels_)
plt.scatter(k1.cluster_centers_[:,0],k1.cluster_centers_[:,1],c='k',marker='x')

plt.subplot(1,2,2)
plt.scatter(x2[:,0],x2[:,1],c=k2.labels_)
plt.scatter(k2.cluster_centers_[:,0],k2.cluster_centers_[:,1],c='r',marker='x')

plt.show()