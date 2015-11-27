import numpy as np


class Distribution(object):
    def get_probability(self):
        raise NotImplementedError()


class DiscreteDistribution(Distribution):
    """
    I am a discrete distribution.
    """

    def __init__(self, n):
        # initialize to 1/n
        self.probabilities = np.matrix([[1. / n] * n] * n)


class AbstractHMM(object):
    """
    This is an abstract HMM class which has to be inherited
    by all other HMM classes. It defines a generic HMM.
    """

    def __init__(self, number_of_states, state_labels=None):
        if state_labels is None:
            state_labels = ["State_%d" % i for i in range(number_of_states)]
        assert len(set(state_labels)) == number_of_states
        self.nStates = number_of_states
        self.transitionMatrix = np.matrix([[1. / self.nStates] * self.nStates] * self.nStates)
        self.initialProbabilities = np.matrix([[1. / self.nStates] * self.nStates] * self.nStates)
        self.emissionDistributions = [None for _ in range(self.nStates)]
        self.stateLabels = state_labels

    def transition_probability(self, origin, destination):
        """
        Returns the probability of transitioning from the origin state to destination state.
        Uses state labels.
        :param origin:
        :param destination:
        :return:
        """
        return self._transition_probability(self.stateLabels.index(origin),
                                            self.stateLabels.index(destination))

    def _transition_probability(self, origin, destination):
        pass
