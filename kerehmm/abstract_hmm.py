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
        self.transitionMatrix = np.asmatrix(np.empty((self.nStates, self.nStates)))
        self.transitionMatrix.fill(np.log(1. / self.nStates))
        self.initialProbabilities = np.asmatrix(np.empty((self.nStates, self.nStates)))
        self.initialProbabilities.fill(np.log(1. / self.nStates))

        self.emissionDistributions = np.array([None for _ in range(self.nStates)], dtype=object)
        self.stateLabels = state_labels

    def transition_probability(self, origin, destination):
        """
        Returns the log-probability of transitioning from the origin state to destination state.
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

    def train(self, observations):
        """
        This solves the problem 3 in Rabiner 1989, i.e.
        argmax_Model(P(Model|O)).
        :param observations:
        :return:
        """
        raise NotImplementedError()

    def viterbi_path(self, observations):
        """
        This solves the problem 2 in Rabiner 1989, i.e.
        argmax_Q(P(Q,O|Model).
        :param observations:
        :return:
        """
        raise NotImplementedError()

    def forward_probability(self, state, observations):
        """
        This solves the problem 1 in Rabiner 1989, i.e.
        P(O|Model), equations 19, 20 and 21.
        :param state:
        :param observations:
        :return:
        """
        alpha = np.zeros(shape=len(observations))
        alpha[0] = self.initial_probability(state) * self.emission_probability(state, observations[0])
        for i in range(1, len(observations)):
            for _state in range(self.nStates):

        raise NotImplementedError()

    def backward_probability(self, observations):
        raise NotImplementedError()

    def emission_probability(self, state, observation):
        return self.emissionDistributions[state].get_probability(observation)
