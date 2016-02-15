import numpy as np
from scipy.stats import multivariate_normal

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
