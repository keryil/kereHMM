import numpy as np
from numpy.random import choice
from scipy.stats import multivariate_normal

from kerehmm.util import random_simplex


class Distribution(object):
    def get_probability(self, observation):
        raise NotImplementedError()

    def __getitem__(self, item):
        return self.get_probability(item)

    def emit(self):
        """
        Emits a randomly drawn output.
        :return:
        """
        raise NotImplementedError()

class DiscreteDistribution(Distribution):
    """
    I am a discrete distribution.
    """

    def __init__(self, n, randomize=False):
        self.n = n
        # initialize to 1/n
        self.probabilities = np.log(np.array([1. / n] * n))
        if randomize:
            self.probabilities = np.log(random_simplex(n))

    def get_probability(self, observation):
        return self.probabilities[observation]

    def emit(self):
        return choice(range(self.n), p=np.exp(self.probabilities))


class GaussianMixture(Distribution):
    """
    I am a mixture of continuous gaussians.
    """

    def __init__(self, nmixtures, dimensions):
        self.nmixtures = nmixtures
        self.dimensions = dimensions
        self.means = np.zeros(shape=(nmixtures, dimensions))
        self.variances = np.zeros(shape=(nmixtures, dimensions, dimensions))
        map(lambda x: np.fill_diagonal(x, 1), self.variances)
        self.weights = np.zeros(shape=(nmixtures,))
        self.weights[:] = np.log(1. / nmixtures)

    def get_probability(self, observation):
        running_sum = -np.inf
        for i, component in enumerate(self.weights):
            dist = multivariate_normal(mean=self.means[i],
                                       cov=self.variances[i])
            prob = np.log(np.sum(dist.pdf(observation)))
            running_sum = np.logaddexp(component + prob,
                                       running_sum)
        return running_sum

    def emit(self):
        mixture = choice(range(self.nmixtures), p=np.exp(self.weights))
        dist = multivariate_normal()
        return dist.rvs(mean=self.means[mixture], cov=self.variances[mixture])
