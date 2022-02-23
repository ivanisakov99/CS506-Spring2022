from collections import defaultdict
from math import inf
import numpy as np


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)

    Returns a new point which is the center of all the points.
    """
    centre = [0] * len(points[0])

    for i in range(len(points)):
        for j in range(len(points[i])):
            centre[j] += points[i][j]

    for i in range(len(centre)):
        centre[i] /= len(points)

    return centre


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    clusters, centres = defaultdict(list), defaultdict(list)

    for i, cluster in enumerate(assignments):
        clusters[cluster].append(dataset[i])

    for i, points in clusters.items():
        centres[i] = point_avg(points)

    return [centre for centre in centres.values()]


def assign_points(data_points, centers):
    """
    Accepts a list of data points and a list of centres.
    Assigns each data point to its closest centre.
    Returns a list of centre assignments for each point.
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between `a` and `b`
    """
    res = 0
    for i in range(len(a)):
        res += (a[i] - b[i])**2
    return res**(1/2)


def distance_squared(a, b):
    return distance(a, b) ** 2


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    index = np.random.choice(len(dataset), k, replace=False)
    return [dataset[i] for i in index]


def cost_function(clustering: defaultdict[:, list]):
    """
    The cost function for K-Means.
    ->`∑dist(x_i, C_i)` for `k` clusters.
    """
    result = 0

    for _, points in clustering.items():
        centre = point_avg(points)
        for point in points:
            result += distance(centre, point)

    return result


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    centres = []
    available_points = set(tuple(data) for data in dataset)

    centres.append(dataset[np.random.randint(len(available_points))])
    available_points.remove(tuple(centres[-1]))

    for _ in range(k - 1):
        distances = list()

        for point in available_points:
            distances.append([distance_squared(centres[-1], point), point])

        distances.sort()
        random = np.random.random()
        r = 0.0

        for entry in distances:
            r += entry[0]

            if r > random:
                centres.append(list(entry[1]))
                available_points.remove(entry[1])
                break

    return centres


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset)+1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
