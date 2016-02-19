import numpy as np


def add_logs(lst):
    """
    Adds a list of log-scale values in decimal scale and converts them back.
    :param lst:
    :return:
    """
    return reduce(np.logaddexp, lst)


DELTA_P = 1.0e-10


def random_simplex(size, two_d=False):
    """
    Returns a random simplex of given shape
    :param shape:
    :return:
    """
    if two_d:
        arr = np.random.rand(size, size)
        for i in range(size):
            arr[i, i:] = arr[i, i:] / np.sum(arr[i, i:]) * (1 - np.sum(arr[i, :i]))
            arr[i + 1:, i] = arr[i + 1:, i] / np.sum(arr[i + 1:, i]) * (1 - np.sum(arr[:i + 1, i]))
    else:
        arr = np.random.random(size)
        arr /= np.sum(arr)
    return arr
