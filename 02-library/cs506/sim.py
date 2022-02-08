import numpy as np
from cmath import sqrt


def euclidean_dist(x, y):
    """
    """
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)


def manhattan_dist(x, y):
    """
    """
    return sum([abs(x[i] - y[i]) for i in range(len(x))])


def jaccard_dist(x, y):
    """
    """
    if len(x) == 0 and len(y) == 0:
        return 0

    intersection = len(set(x).intersection(y))
    union = (len(set(x)) + len(set(y))) - intersection
    return 1 - intersection / union


def cosine_sim(x, y):
    """
    """
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        return 0
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# Feel free to add more
