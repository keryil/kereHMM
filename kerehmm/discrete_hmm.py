import numpy as np


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
        :param origin:
        :param destination:
        :return:
        """
        return self.transitionMatrix[origin, destination]

    def l2s(self, label):
        """
        Converts label to state index.
        :rtype: str
        :param label: state label
        :return: state index
        """
        return self.stateLabels.index(label)

    def s2l(self, state):
        """
        Converts state index to label
        :rtype: int
        :param state: state index
        :return: state label
        """
        return self.stateLabels[state]

    def initial_probability(self, state):
        """

        :param state:
        :return:
        """
        return self.initialProbabilities[state]

    def emission_probability(self, state, emission):
        """

        :param state:
        :param emission:
        :return:
        """
        return self.emissionDistributions[state].get_probability(emission)
