from collections import namedtuple

import numpy as np
from numpy.random import choice

from kerehmm.util import DELTA_P, random_simplex, CONVERGENCE_DELTA_LOG_LIKELIHOOD


class AbstractHMM(object):
    """
    This is an abstract HMM class which has to be inherited
    by all other HMM classes. It defines a generic HMM.
    """

    def __init__(self, number_of_states, state_labels=None, verbose=False, random_transitions=False,
                 *args, **kwargs):
        self.current_state = None
        if state_labels is None:
            state_labels = ["State_%d" % i for i in range(number_of_states)]
        assert len(set(state_labels)) == number_of_states
        self.nStates = number_of_states
        self.transitionMatrix = np.empty((self.nStates, self.nStates))
        self.initialProbabilities = np.empty(shape=self.nStates)
        if not random_transitions:
            self.transitionMatrix.fill(np.log(1. / self.nStates))
            self.initialProbabilities.fill(np.log(1. / self.nStates))
        else:
            self.transitionMatrix = np.log(random_simplex(self.nStates, two_d=True))
            self.initialProbabilities = np.log(random_simplex(self.nStates))

        self.emissionDistributions = np.array([None for _ in range(self.nStates)], dtype=object)
        self.stateLabels = state_labels
        self.verbose = verbose

    def sanity_check(self, sanitize=False, verbose=None):
        """
        Checks if the parameters are in range, and if the distributions provided are true probability distributions.
        If @sanitize=True, the method attempts to sanitize improper parameters.
        :return:
        """
        _tmp_verbose = self.verbose
        if verbose is not None:
            self.verbose = verbose

        assert np.isclose(np.logaddexp.reduce(self.initialProbabilities), np.log(1))
        if self.verbose:
            print "SANITY CHECK: transmat\n{}".format(np.exp(self.transitionMatrix))
        for (row, _), (column, _) in zip(enumerate(self.transitionMatrix), enumerate(self.transitionMatrix.T)):
            sum1 = np.logaddexp.reduce(self.transitionMatrix[row])
            sum2 = np.logaddexp.reduce(self.transitionMatrix[:, column])
            if self.verbose:
                print "SANITY CHECK #{}: Row sum: {}, Column sum: {}".format(row, np.exp(sum1), np.exp(sum2))
            try:
                assert np.isclose(sum1, np.log(1))
            except AssertionError, e:
                print "Offending row: {} (sums up to {})".format(np.exp(self.transitionMatrix[row]),
                                                                 np.sum(np.exp(self.transitionMatrix[row])))
                if not sanitize:
                    raise e
                else:
                    self.transitionMatrix -= np.logaddexp.reduce(self.transitionMatrix, axis=1)[:, np.newaxis]
                    print "Corrected to: {}".format(np.exp(self.transitionMatrix[row]))
        self.verbose = _tmp_verbose

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
        initial_emissions = self.emission_probability(state=None, observation=observations[0])
        print "Initial emission probs: {}".format(np.exp(initial_emissions))
        print "Initial state probs: {}".format(np.exp(self.initialProbabilities))
        alpha[0,] = self.initialProbabilities + initial_emissions
        # scaling_parameters = np.empty(shape=len(observations))

        for t, obs in enumerate(observations):
            if t == 0:
                continue
            for state in range(self.nStates):
                transitions = np.empty(shape=(self.nStates,))
                transitions[:] = alpha[t - 1,] + self.transitionMatrix[:, state]
                alpha[t, state] = np.logaddexp.reduce(transitions) + self.emission_probability(state, obs)
        if self.verbose:
            print "alpha = %s" % np.exp(alpha)
            print "***********"
        return alpha

    def backward_probability(self, observations):
        raise NotImplementedError()

    def gamma2(self, observations):
        """
        Equation 27 from Rabiner 1989. Utilised for training.

        :param observations:
        :return:
        """
        alpha = self.forward(observations)
        beta = self.backward(observations)
        print "Alpha", np.exp(alpha)
        print "Beta", np.exp(beta)
        alpha_times_betas = (alpha + beta)
        print "Product", np.exp(alpha_times_betas)
        denom = np.logaddexp.reduce(alpha_times_betas)
        print "Denominator", np.exp(denom)
        denom_expanded = np.repeat(np.atleast_2d(denom), len(observations), axis=0)
        print "Expanded Denominator", denom_expanded
        gamma = alpha_times_betas - denom_expanded

        print "GAMMA", np.exp(gamma)
        return gamma

    def gamma3(self, alpha, beta):
        gamma = alpha + beta - np.expand_dims(np.logaddexp.reduce(alpha + beta, axis=-1), axis=1)
        return gamma

    def gamma(self, xi=None, observations=None):
        """
        Equation 38 from Rabiner 1989.
        :param observations:
        :return:
        """
        if xi is None:
            assert observations
            xi = self.xi(observations)
        alpha = self.forward(observations)
        beta = self.backward(observations)
        gamma = np.logaddexp.reduce(xi, axis=-1)
        # print np.exp(gamma)
        # append last line
        prod = alpha[-1,] + beta[-1,]
        prod = prod - np.logaddexp.reduce(prod)
        # print np.exp(prod)
        gamma = np.vstack((gamma, prod))
        gamma = np.empty(shape=(len(xi), self.nStates))
        # for t, matrix in enumerate(xi):
        #     x = np.logaddexp.reduce(matrix[:], axis=1)[0]
        #     gamma[t] = x
        return gamma

    def xi(self, observations, alpha=None, beta=None):
        """
        Equation 37 from Rabiner 1989. Utilised for training.
        :param observations:
        :return:
        """
        from itertools import product
        xi = np.empty(shape=(len(observations) - 1, self.nStates, self.nStates))
        if alpha is not None:
            alpha = self.forward(observations)
        if beta is not None:
            beta = self.backward(observations)

        for t, _ in enumerate(observations[:-1]):
            running_sum = -np.inf
            for i, j in product(range(self.nStates), range(self.nStates)):
                xi[t, i, j] = alpha[t, i] \
                              + self.transitionMatrix[i, j] \
                              + self.emission_probability(j, observations[t + 1]) \
                              + beta[t + 1, j]
                # running_sum = np.logaddexp(running_sum, xi[t, i, j])
            xi[t, :] -= np.logaddexp.reduce(alpha[-1])
        return xi

    def backward(self, observations):
        """
        Computes the backward probabilities of a string of observations, as
        in Rabiner 1989's equations 23, 24, and 25.
        :param observations:
        :return:
        """
        # beta[time, state]
        beta = np.empty(shape=(len(observations), self.nStates))
        # this used to be log(1), but I changed it to mimic ghmm library.
        beta[-1,] = np.log(1.0)  # / self.nStates)

        for t in range(0, len(observations) - 1)[::-1]:
            # print beta
            for i in range(self.nStates):
                transitions = np.empty(shape=self.nStates)
                transitions[:] = self.transitionMatrix[i, :] + \
                                 self.emission_probability(None,
                                                           observations[t + 1])

                beta[t, i] = np.logaddexp.reduce(transitions + beta[t + 1,])
        # beta[-1, ] = np.log(1.0 / self.nStates)
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
            # print "Emission dists: {}".format(self.emissionDistributions)
            # print "Observation: {}".format(observation)
            # print "Prob: {}".format(np.exp(self.emissionDistributions[0][observation]))
            return np.array([float(self.emissionDistributions[s][observation]) for s in range(self.nStates)])
        return self.emissionDistributions[state][observation]

    def setup_strict_left_to_right(self, set_emissions=False):
        """
        Converts this HMM into a strictly left-to-right one.
        :return:
        """
        delta_p = DELTA_P
        self.transitionMatrix[:] = np.log(delta_p)
        for state in range(self.nStates):
            try:
                self.transitionMatrix[state, state + 1] = np.log(1 - delta_p * (self.nStates - 1))
            # last state cannot have a valid trans[state+1]
            except IndexError:
                assert state == self.nStates - 1
                self.transitionMatrix[state, 0] = np.log(1 - delta_p * (self.nStates - 1))

        # set the initial probs the same way
        self.initialProbabilities = np.array([np.log(1 - delta_p * (self.nStates - 1))] + \
                                             [np.log(delta_p)] * (self.nStates - 1))

        if set_emissions:
            # set emission probabilities so that each state only emits
            # its own label
            for i, (state, distribution) in enumerate(zip(range(self.nStates), self.emissionDistributions)):
                probs = np.empty_like(distribution.probabilities)
                probs[:] = -np.inf
                probs[i] = 0
                distribution.probabilities = probs

    def do_pass(self, observations):
        raise NotImplementedError()

    def train(self, observations, iterations=None, auto_stop=True, verbose=False):
        """
        This is the training algorithm in Rabiner 1989
        equations 40a, 40b and 40c.
        :param observations:
        :param iterations:
        :return:
        """
        if auto_stop and not iterations:
            i = 0
            while True:
                i += 1
                old_f = self.forward_probability(observations)
                self.do_pass(observations, not (i % 100))
                if not (i % 100):
                    print 'ITERATION #{}'.format(i)
                    print 'Log likelihood = {}'.format(old_f)
                    print 'Delta llk = {}'.format(delta_p)
                if auto_stop:
                    delta_p = self.forward_probability(observations) - old_f
                    if delta_p <= CONVERGENCE_DELTA_LOG_LIKELIHOOD:
                        break
        else:
            for i in range(iterations + 1):
                old_f = self.forward_probability(observations)
                self.do_pass(observations, verbose)
                if auto_stop:
                    delta_p = self.forward_probability(observations) - old_f
                    # if not (i % 100):
                    print 'ITERATION #{}'.format(i)
                    print 'Log likelihood = {}'.format(old_f)
                    print 'Delta llk = {}'.format(delta_p)
                    if delta_p <= CONVERGENCE_DELTA_LOG_LIKELIHOOD:
                        break

        print "Finished training at {} iterations.".format(i)
        print "DELTA FORWARD LOG PROB AFTER TRAINING:", delta_p
        print "{} --> {}".format(np.exp(old_f), np.exp(self.forward_probability(observations)))
        # print delta_p / np.exp(old_f)

    def emit(self):
        return self.emissionDistributions[self.current_state].emit()

    def transition(self):
        if self.current_state:
            self.current_state = choice(range(self.nStates), p=np.exp(self.transitionMatrix[self.current_state,]))
        else:
            self.current_state = choice(range(self.nStates), p=np.exp(self.initialProbabilities))

    def simulate(self, iterations=1, reset=True):
        emissions = []
        states = []
        if reset:
            self.current_state = None

        for i in range(iterations):
            self.transition()
            states.append(self.current_state)
            emissions.append(self.emit())

        return namedtuple("SimulationResult", ['states', 'emissions'])(states=states, emissions=emissions)
