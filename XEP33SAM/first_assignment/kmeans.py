import numpy as np
from pyflann import *
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans
import time
import warnings

warnings.filterwarnings('ignore')


def sum_of_distances(points, closest, centroids):
    """
    Calculate sum of distances between the points and the centroids
    """
    dist = 0
    for j in range(points.shape[0]):
        dist += (points[j, :] - centroids[closest[j]])**2

    return np.sum(dist)


def initialize_centroids(points: np.ndarray, k: int, seed: int=1234) -> np.ndarray:
    """Returns k random centroids from the initial points
    :type k: int
    :type seed: int
    """
    c = points.copy()
    np.random.seed(seed)
    np.random.shuffle(c)
    return c[:k]


# def closest_centroid(points, centroids):
#    """
#    Returns an array containing the index to the nearest centroid for each point.
#    This is used on the exact distance k-means only.
#    """
#    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
#    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids, seed: int=1234):
    """
    Returns the new centroids assigned from the points closest to them.
    """
    new_centroids = []
    for k in range(centroids.shape[0]):
        # Treat the case where a centroid has no points assigned
        # by selecting a random point as a new centroid
        if len(points[closest == k]) > 0:
            new_centroids.append(points[closest == k].mean(axis=0))
        else:
            centroids = points.copy()
            np.random.seed(seed)
            idx = np.random.randint(len(points))
            new_centroids.append(points[idx])

    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])


num_iterations = 30
K = 100

# Read the data and shape them accordingly
vec = np.fromfile('data/oxford_5k/imagedesc.dat', dtype=np.float32)
points = vec.reshape(-1, 128)
# vec2 = np.fromfile('data/SIFT.dat', dtype=np.uint8)
# points = np.float32(vec2.reshape(-1, 128))


# names = pd.read_csv('data/oxford_5k/imagenames.txt', header=None)

init_centroids = initialize_centroids(points, K)


# K-means using randomized k-d tree for distance between points and centroids
start_time = time.time()
print("")
print("****************************************")
print("Starting approximate k-means with k-d trees (90% target precision)")
centroids = init_centroids
flann1 = FLANN()

for i in range(num_iterations):
    params = {'algorithm': 'kdtree',
              'trees': 10,
              'target_precision': 0.90,
              'dtype': 'euclidean'}

    closest, dists = flann1.nn(centroids, points, num_neighbors=1, **params)
    centroids = move_centroids(points, closest, centroids)

print("Distance based on flann:", np.sum(dists))
distance = sum_of_distances(points, closest, centroids)
print("Total distance from assigned centroids:", distance)
print("Execution time: %s seconds: " % (time.time() - start_time))

start_time = time.time()
# K-means using k-means search trees for distance between points and centroids
print("")
print("****************************************")
print("Starting approximate k-means with k-means search tree (90% target precision)")
centroids = init_centroids
flann2 = FLANN()
for _ in range(num_iterations):
    params = {'algorithm': 'kmeans',
              'branching': 64,
              'target_precision': 0.90,
              'dtype': 'euclidean'
              }
    closest, dists = flann2.nn(centroids, points, num_neighbors=1, **params)
    centroids = move_centroids(points, closest, centroids)

print("Distance based on flann:", np.sum(dists))
print("Total distance from assigned centroids:", sum_of_distances(points, closest, centroids))
print("Execution time: %s seconds: " % (time.time() - start_time))

# K-means using exact distance
start_time = time.time()
print("****************************************")
print("Starting k-means with linear distance")
centroids = init_centroids
#kmeans = KMeans(n_clusters=K,
#                init=centroids,
#                algorithm='full',
#                n_jobs=3,
#                max_iter=num_iterations).fit(points)

# for i in range(num_iterations):
#    #closest = closest_centroid(points, centroids)
#    closest = pairwise_distances_argmin(points, centroids)
#    centroids = move_centroids(points, closest, centroids)
flann3 = FLANN()
for _ in range(num_iterations):
    params = {'algorithm': 'linear',
              'dtype': 'euclidean'
              }
    closest, dists = flann3.nn(centroids, points, num_neighbors=1, **params)
    centroids = move_centroids(points, closest, centroids)

print("Distance based on flann:", np.sum(dists))
print("Total distance from assigned centroids:", sum_of_distances(points, closest, centroids))


# print("Total distance from assigned centroids:", kmeans.inertia_)
# print(closest[0])
print("Execution time: %s seconds: " % (time.time() - start_time))

# print("Shape of centroids: ", centroids.shape)
# print("Shape of extended centroids: ", centroids[:, np.newaxis].shape)
# print("Shape of points: ", points.shape)
# print("Shape of closest: ", closest.shape)
