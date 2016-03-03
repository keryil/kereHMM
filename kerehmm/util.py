import numpy as np

CONVERGENCE_DELTA_LOG_LIKELIHOOD = 1e-05
DELTA_P = 1.0e-10


def add_logs(lst):
    """
    Adds a list of log-scale values in decimal scale and converts them back.
    :param lst:
    :return:
    """
    return reduce(np.logaddexp, lst)


def normalize(a):
    for i, _ in enumerate(a):
        a[i, i:] = a[i, i:] / np.sum(a[i, i:]) * (1 - np.sum(a[i, :i]))
        a[i + 1:, i] = a[i + 1:, i] / np.sum(a[i + 1:, i]) * (1 - np.sum(a[:i + 1, i]))
    return a


def random_simplex(size, two_d=False, log_scale=False):
    """
    Returns a random simplex of given shape
    :param shape:
    :return:
    """
    if two_d:
        arr = np.random.uniform(low=DELTA_P, high=.9, size=size ** 2).reshape((size, size))
    else:
        arr = np.random.uniform(low=DELTA_P, high=.9, size=size)
    # make differences more extreme
    # arr = arr * arr
    arr = smooth_probabilities(arr)
    return arr if not log_scale else np.log(arr)


def smooth_probabilities(probabilities):
    """
    Removes zero entries and replaces them with DELTA_P, and making sure
    we still have a proper simplex that adds up to one at each row.
    :param probabilities:
    :return:
    """
    if len(probabilities.shape) == 2:
        for i, _ in enumerate(probabilities):
            counter = 0
            probabilities[i] /= probabilities[i].sum()
            if (probabilities[i] < DELTA_P).any():
                zero_items = probabilities[i][probabilities[i] < DELTA_P]
                counter += zero_items
                probabilities[i] -= counter * DELTA_P / (len(probabilities[i]) - zero_items)
                probabilities[i][probabilities[i] < DELTA_P] = DELTA_P
            probabilities[i] /= probabilities[i].sum()
    elif len(probabilities.shape) == 1:
        counter = len(probabilities[probabilities < DELTA_P])
        probabilities -= counter * DELTA_P / (len(probabilities) - counter)
        probabilities[probabilities < DELTA_P] = DELTA_P
        probabilities /= probabilities.sum()
    else:
        raise ValueError("Expected a 1d or 2d array, gotten shape {}.".format(probabilities.shape))
    return probabilities
