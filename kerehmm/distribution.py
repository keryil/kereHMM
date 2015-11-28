import numpy as np


class Distribution(object):
    def get_probability(self, observation):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.get_probability(item)

class DiscreteDistribution(Distribution):
    """
    I am a discrete distribution.
    """

    def __init__(self, n):
        self.n = n
        # initialize to 1/n
        self.probabilities = np.log(np.array([1. / n] * n))

    def get_probability(self, observation):
        return self.probabilities[observation]
