from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
n_digits = 10
points = np.array([[-4.58959238, -0.8432033 ],
       [ 1.71636211, -1.47923651],
       [ 4.95071736, -0.68965574],
       [ 1.21000187, -3.93779568],
       [ 1.95884528,  1.52057292],
       [ 3.8878771 ,  5.31429289],
       [-1.91174901, -1.37051631],
       [-0.01205497,  3.52017988],
       [-0.44401107,  0.58105961],
       [-2.71000273,  1.69218605]])
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(points)
centroids = kmeans.cluster_centers_

h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(points[:, 0], points[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X

plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('[Sklearn]K-means clustering on the digits dataset\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()