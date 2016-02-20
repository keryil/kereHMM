import numpy as np
from numpy.random import choice
from scipy.stats import multivariate_normal, norm

from kerehmm.util import random_simplex


class Distribution(object):
    def get_probability(self, observation, *args, **kwargs):
        raise NotImplementedError()

    def __getitem__(self, item, *args, **kwargs):
        return self.get_probability(item, *args, **kwargs)

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

    def get_probability(self, observation, *args, **kwargs):
        return self.probabilities[observation]

    def emit(self):
        return choice(range(self.n), p=np.exp(self.probabilities))


class ContinuousDistribution(Distribution):
    """
    I am a gaussian distribution.

    It is initialized to mean 0 across dimensions, and
    the covariance is set to a scalar matrix with 1
    across the diagonal.
    >>> m = ContinuousDistribution(2)
    >>> m.mean
    array([ 0.,  0.])
    >>> m.variance
    array([[ 1.,  0.],
           [ 0.,  1.]])

    You can simply use bracket indexing to get the probability of
    an observation. Note that all probabilities are in log scale.
    >>> from scipy import stats
    >>> m[(0, 0)] == stats.multivariate_normal(mean=m.mean, cov=m.variance).logpdf((0,0))
    True

    It also supports univariate gaussians.
    >>> m = ContinuousDistribution(1)
    >>> m.mean
    0
    >>> m.variance
    1
    >>> m[0] == stats.norm(loc=0, scale=1).logpdf(0)
    True
    """
    def __init__(self, dimensions, random=False):
        self.dimensions = dimensions

        def mean():
            if random:
                return np.random.uniform(-10, 10, size=self.dimensions)
            else:
                if self.dimensions == 1:
                    return 0
                else:
                    return np.zeros(shape=(self.dimensions,))

        def variance():
            if random:
                variances = np.zeros((self.dimensions, self.dimensions))
                np.fill_diagonal(variances, 1)
                return variances
            else:
                if self.dimensions == 1:
                    return 1
                else:
                    variances = np.zeros((self.dimensions, self.dimensions))
                    np.fill_diagonal(variances, 1)
                    return variances

        self.mean = mean()
        self.variance = variance()

    def get_probability(self, observation, *args, **kwargs):
        if self.dimensions != 1:
            return multivariate_normal(mean=self.mean, cov=self.variance).logpdf(observation)
        else:
            return norm(loc=self.mean, scale=self.variance).logpdf(observation)

    def emit(self):
        if self.dimensions != 1:
            dist = multivariate_normal(mean=self.mean, cov=self.variance)
        else:
            dist = norm(loc=self.mean, scale=self.variance)
        return dist.rvs()

class GaussianMixture(Distribution):
    """
    I am a mixture of continuous gaussians.

    A single component mixture is essentially a Gaussian.
    It is initialized to mean 0 across dimensions, and
    the covariance is set to a scalar matrix with 1
    across the diagonal.
    >>> m = GaussianMixture(1, 2)
    >>> m.means
    array([[ 0.,  0.]])
    >>> m.variances
    array([[[ 1.,  0.],
            [ 0.,  1.]]])

    You can simply use bracket indexing to get the probability of
    an observation. Note that all probabilities are in log scale.
    >>> from numpy import exp
    >>> exp(m[(0, 0)])
    0.15915494309189535
    >>> exp(m[(1, 2)])
    0.013064233284684921
    """

    def __init__(self, nmixtures, dimensions):
        self.nmixtures = nmixtures
        self.dimensions = dimensions
        self.means = np.zeros(shape=(nmixtures, dimensions))
        self.variances = np.zeros(shape=(nmixtures, dimensions, dimensions))
        map(lambda x: np.fill_diagonal(x, 1), self.variances)
        self.weights = np.zeros(shape=(nmixtures,))
        self.weights[:] = np.log(1. / nmixtures)

    def get_probability(self, observation, *args, **kwargs):
        running_sum = -np.inf
        for weight, mean, variance in zip(self.weights, self.means, self.variances):
            dist = multivariate_normal(mean=mean,
                                       cov=variance)
            prob = np.log(np.sum(dist.pdf(observation)))
            running_sum = np.logaddexp(weight + prob,
                                       running_sum)
        return running_sum

    def emit(self):
        mixture = choice(range(self.nmixtures), p=np.exp(self.weights))
        dist = multivariate_normal()
        return dist.rvs(mean=self.means[mixture], cov=self.variances[mixture])
