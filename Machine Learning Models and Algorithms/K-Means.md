k means
================
Haolin Zhong (UNI: hz2771)
2021/10/4

# Core idea

-   User should select k initial centroids for cluster; Keep recursion
    until clusters converges
-   Process:
    -   Pick k initial centroids
    -   Repeat:
        -   for every sample point, select its nearest centroids as its
            cluster
        -   for each cluster, re-calculate the centroid
    -   Until: centroids stay the same or reach maximum recursion

# Practice

## Import dependencies

``` python
import numpy as np
import matplotlib.pyplot as plt

# generate cluster data from sklearn
from sklearn.datasets.samples_generator import make_blobs
```

## Load and inspect data

``` python
X, Y = make_blobs(n_samples = 100, centers = 6, random_state=1234, cluster_std=0.6)

plt.close()
plt.figure(figsize= (6,6))
plt.scatter(X[:,0], X[:, 1], c=Y)
plt.show()
```

<img src="K-Means_files/figure-gfm/unnamed-chunk-2-1.png" width="576" />

## Algorithm implementation

``` python
from scipy.spatial.distance import cdist

class K_means(object):
    def __init__(self, n_clusters=6, max_iter=300, centroids = []):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = np.array(centroids, dtype=np.float)
        
    def fit(self, data):
        # The process of cluster
        # If no initial centroid is given, randomly draw from data
        if(self.centroids.shape == (0,)):
            self.centroids = data[np.random.randint(0, data.shape[0], self.n_clusters), :]
            
        for i in range(self.max_iter):
            # calculate distance to every centroid for every point
            distances = cdist(data, self.centroids)
            
            # choose closest centroid as cluster
            c_ind = np.argmin(distances, axis = 1)
            
            # update the coordinate of centroids
            for i in range(self.n_clusters):
                # exclude the cluster(s) which does not appear in c_ind
                for i in c_ind:
                    self.centroids[i] = np.mean(data[c_ind == i], axis = 0)
    
    def predict(self, X):
        distances = cdist(X, self.centroids)
        return np.argmin(distances, axis = 1)
            
```

## Test

``` python
def plot_Kmeans(x, y, centroids, subplot, title):
    plt.subplot(subplot)
    plt.scatter(x[:,0], x[:,1], c='r')
    plt.scatter(centroids[:,0], centroids[:,1], c=np.array(range(6)), s=100)
    plt.title(title)
    
kmeans = K_means(centroids=np.array([[2,1],[2,2],[2,3],[2,4],[2,5],[2,6]]))

plt.close()
plt.figure(figsize=(16,6))
plot_Kmeans(X, Y, kmeans.centroids, 121, 'initial status')

kmeans.fit(X)

plot_Kmeans(X, Y, kmeans.centroids, 122, 'final status')

plt.show()
```

<img src="K-Means_files/figure-gfm/unnamed-chunk-4-3.png" width="1536" />

``` python
X_test = np.array([[0,0], [10, 7]])
Y_pred = kmeans.predict(X_test)

print(Y_pred)
```

    ## [1 5]

``` python
print(kmeans.centroids)
```

    ## [[ 5.76444812 -4.67941789]
    ##  [-2.89174024 -0.22808556]
    ##  [-5.89115978  2.33887408]
    ##  [-4.53406813  6.11523454]
    ##  [-1.15698106  5.63230377]
    ##  [ 9.20551979  7.56124841]]
