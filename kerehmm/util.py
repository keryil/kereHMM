import numpy as np


def add_logs(lst):
    """
    Adds a list of log-scale values in decimal scale and converts them back.
    :param lst:
    :return:
    """
    return reduce(np.logaddexp, lst)


DELTA_P = 1.0e-10


def normalize(a):
    for i, _ in enumerate(a):
        a[i, i:] = a[i, i:] / np.sum(a[i, i:]) * (1 - np.sum(a[i, :i]))
        a[i + 1:, i] = a[i + 1:, i] / np.sum(a[i + 1:, i]) * (1 - np.sum(a[:i + 1, i]))
    return a


def random_simplex(size, two_d=False):
    """
    Returns a random simplex of given shape
    :param shape:
    :return:
    """
    if two_d:
        def new():
            a = np.random.uniform(low=.1, high=1. / size, size=(size, size))
            for i in range(size):
                a[i, i:] = a[i, i:] / np.sum(a[i, i:]) * (1 - np.sum(a[i, :i]))
                a[i + 1:, i] = a[i + 1:, i] / np.sum(a[i + 1:, i]) * (1 - np.sum(a[:i + 1, i]))
            return a

        arr = new()
        while (arr <= 0).any():
            arr = new()

            # print "Returning {}".format(arr)
    else:
        arr = np.random.uniform(low=DELTA_P, high=1, size=size)
        arr /= np.sum(arr)
    return arr


CONVERGENCE_DELTA_LOG_LIKELIHOOD = 1e-05
