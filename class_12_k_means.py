# class 12th = 2022, 29th Jan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = -2 * np.random.rand(100, 2)
X1 = 1 + 2 * np.random.rand(50, 2)
X[50:100, :] = X1

plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.show()

Kmean = KMeans(n_clusters=2)
Kmean.fit(X)

print(Kmean.cluster_centers_)

plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.scatter(-0.94665068, -0.97138368, s=200, c='g', marker='s')
plt.scatter(2.01559419, 2.02597093, s=200, c='r', marker='s')
plt.show()

print(Kmean.labels_)

sample_test = np.array([-3.0, -3.0])
second_test = sample_test.reshape(1, -1)

print(Kmean.predict(second_test))


from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the Iris dataset with pandas
dataset = pd.read_csv('./data/Iris.csv')


def clustering():
    x = dataset.iloc[:, [1, 2, 3, 4]].values

    # Finding the optimum number of clusters for k-means classification
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++',
                        max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    # Plotting the results onto a line graph, allowing us to observe 'The elbow'
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()

    ######
    # Silhouette plots, another method used to select the optimal k
    # k-means++, a variant of k-means, that improves clustering results through more clever seeding of the initial cluster centers.
    ######

    kmeans = KMeans(
        n_clusters=3, init='k-means++', max_iter=300, n_init=6, random_state=0)
    y_kmeans = kmeans.fit_predict(x)

    # Visualising the clusters
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Iris-setosa')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],
                s=100, c='blue', label='Iris-versicolour')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
                s=100, c='green', label='Iris-virginica')

    # Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                :, 1], s=100, c='yellow', label='Centroids')

    plt.legend()
    plt.show()


clustering()


# references
# - https://towardsdatascience.com/k-means-clustering-from-a-to-z-f6242a314e9a
# - https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
# - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# - https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/

