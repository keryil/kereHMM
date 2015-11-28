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
        self.transitionMatrix = np.empty((self.nStates, self.nStates))
        self.transitionMatrix.fill(np.log(1. / self.nStates))
        self.initialProbabilities = np.empty(shape=self.nStates)
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
        # delta holds the probabilities
        # delta[time, state]
        delta = np.empty(shape=(len(observations), self.nStates))

        # psi holds the best states
        # psi[time, state]
        psi = np.empty(shape=(len(observations), self.nStates))

        initial_emissions = np.array([d[observations[0]] for d in self.emissionDistributions])
        delta[0, :] = self.initialProbabilities + initial_emissions
        psi[0, :] = 0

        for t in range(1, len(observations)):
            for state in range(self.nStates):
                transitions = np.empty(shape=(self.nStates,))
                # probability of transitioning to this state at time t
                # for each state
                transitions[:] = delta[t - 1, :] + self.transitionMatrix[:, state]
                delta[t, state] = np.max(transitions) + \
                                  self.emission_probability(state, observations[t])
                psi[t, state] = np.argmax(transitions)

        # log probability of viterbi path
        path_probability = np.max(delta[-1,])
        path = np.empty_like(observations)
        path[-1] = np.argmax(delta[-1,])
        for t in reversed(range(len(observations) - 1)):
            path[t] = psi[t + 1, path[t + 1]]

        return path, path_probability

    def forward_probability(self, observations):
        """
        This solves the problem 1 in Rabiner 1989, i.e.
        P(O|Model), equations 19, 20 and 21.
        :param observations:
        :return:
        """
        return np.logaddexp.reduce(self.forward(observations)[-1,])

    def forward(self, observations):
        # alpha[time, state]
        alpha = np.empty(shape=(len(observations), self.nStates))
        initial_emissions = np.array([d[observations[0]] for d in self.emissionDistributions])
        alpha[0,] = self.initialProbabilities + initial_emissions

        for t in range(1, len(observations)):
            for state in range(self.nStates):
                transitions = np.empty(shape=(self.nStates,))
                transitions[:] = alpha[t - 1,] + self.transitionMatrix[:, state]
                alpha[t, state] = np.logaddexp.reduce(transitions) + \
                                  self.emission_probability(state, observations[t])
        return alpha

    def backward_probability(self, observations):
        raise NotImplementedError()

    # def gamma(self, observations):
    #     """
    #     Equation 27 from Rabiner 1989. Utilised for training.
    #
    #     :param observations:
    #     :return:
    #     """
    #     alpha_times_beta = self.forward(observations) + self.backward(observations)
    #     gamma = alpha_times_beta - np.logaddexp.reduce(alpha_times_beta, axis=1)
    #     return gamma

    def gamma(self, xi=None, observations=None):
        """
        Equation 38 from Rabiner 1989.
        :param observations:
        :return:
        """
        if xi is None:
            assert observations
            xi = self.xi(observations)

        gamma = np.empty(shape=(len(xi), self.nStates))
        for t, matrix in enumerate(xi):
            x = np.logaddexp.reduce(matrix[:], axis=1)[0]
            print x
            gamma[t] = x
        return gamma

    def xi(self, observations):
        """
        Equation 37 from Rabiner 1989. Utilised for training.
        :param observations:
        :return:
        """
        from itertools import product
        xi = np.empty(shape=(len(observations), self.nStates, self.nStates))
        alpha = self.forward(observations)
        beta = self.backward(observations)
        for t, _ in enumerate(observations[:-1]):
            sum = -np.inf
            for i, j in product(range(self.nStates), range(self.nStates)):
                xi[t, i, j] = alpha[t, i] \
                              + self.transitionMatrix[i, j] \
                              + self.emissionDistributions[j][observations[t + 1]] \
                              + beta[t + 1, j]
                sum = np.logaddexp(sum, xi[t, i, j])
            xi[t, :] = xi[t, :] - sum
        return xi

    def backward(self, observations):
        # beta[time, state]
        beta = np.empty(shape=(len(observations), self.nStates))
        # this is log(1)
        beta[-1,] = 0

        for t in reversed(range(len(observations) - 1)):
            for state in range(self.nStates):
                transitions = np.empty(shape=self.nStates)
                transitions[:] = self.transitionMatrix[state,] + self.emission_probability(None, observations[t + 1]) + \
                                 beta[t + 1,]
                beta[t, state] = np.logaddexp.reduce(transitions)
        return beta

    def emission_probability(self, state, observation):
        """
        Returns the probability of an observation given the state. If the
        state is None, an array containing the probability for each state
        is returned instead.
        :param state:
        :param observation:
        :return:
        """
        if state is None:
            return np.array([self.emissionDistributions[s][observation] for s in range(self.nStates)])
        return self.emissionDistributions[state][observation]

    def setup_strict_left_to_right(self):
        """
        Converts this HMM into a strictly left-to-right one.
        :return:
        """
        delta_p = .00001
        self.transitionMatrix[:] = np.log(delta_p)
        for state in range(self.nStates - 1):
            self.transitionMatrix[state, state + 1] = np.log(1 - delta_p * (self.nStates - 1))

        # set the initial probs the same way
        self.initialProbabilities = np.array([np.log(1 - delta_p * (self.nStates - 1))] + \
                                             [delta_p] * (self.nStates - 1))
