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
            self.transitionMatrix.fill(1. / self.nStates)
            self.initialProbabilities.fill(1. / self.nStates)
        else:
            self.transitionMatrix = random_simplex(self.nStates, two_d=True)
            self.initialProbabilities = random_simplex(self.nStates)

        self.emissionDistributions = np.array([None for _ in range(self.nStates)], dtype=object)
        self.stateLabels = state_labels
        self.verbose = verbose

    def compute_initprob_mapping_score(self, to_hmm, mapping):
        """
        Computes the standardized difference between the mapped initprob
        array and to_hmm.initialProbabilities.

        :param to_hmm:
        :param mapping:
        :return:
        """
        from_prob_init = self.initialProbabilities[mapping]
        to_prob_init = to_hmm.initialProbabilities
        return np.sum(np.abs(to_prob_init - from_prob_init)) / self.nStates

    def compute_transition_mapping_score(self, to_hmm, mapping):
        """
        Computes the standardized difference between the (mapped) transitionMatrix
        array and to_hmm.transitionMatrix.

        :param to_hmm:
        :param mapping:
        :return:
        """
        to_trans = to_hmm.transitionMatrix
        from_trans = self.transitionMatrix[:, mapping][mapping, :]
        return np.sum(np.abs(to_trans - from_trans)) / (self.nStates ** 2)

    def compute_emission_mapping_score(self, to_hmm, mapping):
        raise NotImplementedError("AbstractHMM doesn't define emissions, override this for concrete implementations.")

    def compute_mapping_to(self, to_hmm):
        """
        Computes the most likely mapping from this hmm to the hmm given as a parameter.
        :param hmm:
        :return:
        """
        from itertools import permutations
        from collections import defaultdict
        assert self.nStates == to_hmm.nStates
        nStates = self.nStates
        candidate_mappings = permutations(range(self.nStates))
        scores = dict()
        emission_scores = defaultdict(int)
        for mapping in candidate_mappings:
            # print "Mapping:", mapping
            # compare prob_inits
            scores[mapping] = self.compute_initprob_mapping_score(to_hmm, list(mapping))

            # compare transitions
            scores[mapping] += self.compute_transition_mapping_score(to_hmm, list(mapping))

            # compare emissions
            emission_scores[mapping] = self.compute_emission_mapping_score(to_hmm, list(mapping))

        # normalize emission scores
        emission_scores = np.array(emission_scores.values())
        emission_scores /= np.sum(emission_scores)

        for mapping in candidate_mappings:
            scores[mapping] = scores[mapping] + emission_scores[mapping]

        minimum = min(scores.keys(), key=lambda x: scores[x])
        return minimum

    def rearrange(self, mapping):
        """
        Rearranges the HMM so that the current states map onto the corresponding state
        in the list mapping. For instance, a 3-state hmm with mapping 0,2,1 will have its states
        relabeled as 0->0, 1->2, 2->1. This is useful when comparing HMMs since they
        often have different state orderings.
        :param mapping:
        :return:
        """
        self.initialProbabilities = self.initialProbabilities[mapping]
        self.stateLabels = self.stateLabels[mapping]
        self.transitionMatrix = self.transitionMatrix[mapping, :][:, mapping]
        self.emissionDistributions = self.emissionDistributions[mapping]

    def rearrange_like(self, hmm):
        """
        Rearranges the state ordering of this hmm so that it aligns with the hmm given as the parameter
        as closely as possible.
        :param hmm:
        :return:
        """
        self.rearrange(self.compute_mapping_to(hmm))

    def initialize_parameters(self, observations):
        """
        Initializes (i.e. roughly estimates) the parameters based on the observation before training.
        :param observations:
        :return:
        """
        raise NotImplementedError()

    def sanity_check(self, sanitize=False, verbose=None):
        """
        Checks if the parameters are in range, and if the distributions provided are true probability distributions.
        If @sanitize=True, the method attempts to sanitize improper parameters.
        :return:
        """
        _tmp_verbose = self.verbose
        if verbose is not None:
            self.verbose = verbose

        assert np.isclose(self.initialProbabilities.sum(), 1)
        if self.verbose:
            print "SANITY CHECK: transmat\n{}".format(self.transitionMatrix)
        for row, column in zip(range(self.nStates), range(self.nStates)):
            sum1 = self.transitionMatrix[row].sum()
            sum2 = self.transitionMatrix[:, column].sum()
            if self.verbose:
                print "SANITY CHECK #{}: Row sum: {}, Column sum: {}".format(row, sum1, sum2)
            try:
                assert np.isclose(sum1, 1)
            except AssertionError, e:
                print "Problem in matrix:\n{}".format(self.transitionMatrix)
                print "Offending row: {} (sums up to {})".format(self.transitionMatrix[row],
                                                                 self.transitionMatrix[row].sum())
                if not sanitize:
                    raise e
                else:
                    self.transitionMatrix /= self.transitionMatrix.sum(axis=1)[:, np.newaxis]
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
        delta[0, :] = np.log(self.initialProbabilities) + np.log(initial_emissions)
        psi[0, :] = 0

        for t in range(1, len(observations)):
            for state in range(self.nStates):
                transitions = np.empty(shape=(self.nStates,))
                # probability of transitioning to this state at time t
                # for each state
                transitions[:] = delta[t - 1, :] + np.log(self.transitionMatrix[:, state])
                delta[t, state] = np.max(transitions) + \
                                  np.log(self.emission_probability(state, observations[t]))
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
        log(P(O|Model)), equation 103.
        :param observations:
        :return:
        """
        return -np.logaddexp.reduce(self.forward(observations)[-1])

    def forward(self, observations):
        # alpha[time, state]
        alpha = np.empty(shape=(len(observations), self.nStates))
        scaling_coefficients = np.ones(shape=(len(observations)))

        initial_emissions = self.emission_probability(state=None, observation=observations[0])
        # print "Initial emission probs: {}".format(np.exp(initial_emissions))
        # print "Initial state probs: {}".format(np.exp(self.initialProbabilities))
        alpha[0,] = self.initialProbabilities * initial_emissions
        scaling_coefficients[0] = alpha[0,].sum()
        alpha[0,] /= scaling_coefficients[0]
        # scaling_parameters = np.empty(shape=len(observations))

        for t, obs in enumerate(observations):
            if t == 0:
                continue
            for state in range(self.nStates):
                transitions = np.empty(shape=(self.nStates,))
                transitions[:] = alpha[t - 1,] * self.transitionMatrix[:, state]
                try:
                    alpha[t, state] = transitions.sum() * self.emission_probability(state, obs)
                except ValueError, err:
                    print "t =", t
                    print "self.transitionMatrix[:, {}] = {}".format(state, self.transitionMatrix[:, state])
                    print "alpha[:]\t=\t", alpha
                    print "scale\t=\t", scaling_coefficients
                    print "transitions[:]\t=\t", transitions
                    print "emission_prob({},{})\t=\t".format(state, obs), self.emission_probability(state, obs)
                    raise err
            scaling_coefficients[t] = alpha[t,].sum()
            alpha[t,] /= scaling_coefficients[t]

        if self.verbose:
            print "alpha = %s" % alpha
            print "***********"
        return alpha, scaling_coefficients

    def backward_probability(self, observations):
        raise NotImplementedError()

    # @staticmethod
    def gamma(self, alpha, beta, scale):
        gamma = np.empty(shape=(len(scale), self.nStates))
        for t, c in enumerate(scale):
            gamma[t, :] = alpha[t] * beta[t] * c
        # gamma = (alpha * beta) / np.expand_dims(scale, axis=-1)
        # print gamma
        return gamma

    def zi(self, observations, alpha=None, beta=None, scale=None):
        """
        Equation 37 from Rabiner 1989. Utilised for training.
        :param observations:
        :return:
        """
        from itertools import product
        zi = np.empty(shape=(len(observations) - 1, self.nStates, self.nStates))
        if alpha is not None:
            alpha, scale = self.forward(observations)
        if beta is not None:
            beta = self.backward(observations, scale_coefficients=scale)

        for t, _ in enumerate(observations[:-1]):
            # print -np.log(scale)
            # denom = np.exp(-np.log(scale).sum())
            for i, j in product(range(self.nStates), range(self.nStates)):
                zi[t, i, j] = alpha[t, i] \
                              * self.transitionMatrix[i, j] \
                              * self.emission_probability(j, observations[t + 1]) \
                              * beta[t + 1, j]

                # running_sum = np.logaddexp(running_sum, zi[t, i, j])
                # zi[t, :] /= alpha[-1].sum()
                # zi[t, :] = np.exp(np.log(zi[t, :]) + np.log(scale).sum())
        # print zi
        return zi

    def backward(self, observations, scale_coefficients):
        """
        Computes the backward probabilities of a string of observations, as
        in Rabiner 1989's equations 23, 24, and 25.
        :param observations:
        :return:
        """
        # beta[time, state]
        beta = np.empty(shape=(len(observations), self.nStates))
        # this used to be log(1), but I changed it to mimic ghmm library.
        beta[-1,] = 1.  # / self.nStates)

        for t in range(0, len(observations) - 1)[::-1]:
            # print beta
            for i in range(self.nStates):
                transitions = np.empty(shape=self.nStates)
                # print "Emission probs in forward():", np.exp(self.emission_probability(None, observations[t + 1]))
                transitions[:] = self.transitionMatrix[i, :] \
                                 * self.emission_probability(None,
                                                             observations[t + 1]) \
                                 * beta[t + 1,]

                beta[t, i] = transitions.sum() / scale_coefficients[t + 1]
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
        self.transitionMatrix[:] = delta_p
        for state in range(self.nStates):
            try:
                self.transitionMatrix[state, state + 1] = - delta_p * (self.nStates - 1) + 1.
            # last state cannot have a valid trans[state+1]
            except IndexError:
                assert state == self.nStates - 1
                self.transitionMatrix[state, 0] = - delta_p * (self.nStates - 1) + 1.

        # set the initial probs the same way
        self.initialProbabilities = np.array([1 - delta_p * (self.nStates - 1)] + \
                                             [delta_p] * (self.nStates - 1))

        if set_emissions:
            # set emission probabilities so that each state only emits
            # its own label
            for i, (state, distribution) in enumerate(zip(range(self.nStates), self.emissionDistributions)):
                probs = np.empty_like(distribution.probabilities)
                probs[:] = 0
                probs[i] = 1
                distribution.probabilities = probs

    def do_pass(self, observations):
        raise NotImplementedError()

    def train(self, observations, iterations=100, auto_stop=True, verbose=False):
        """
        This is the training algorithm in Rabiner 1989
        equations 40a, 40b and 40c.
        :param observations:
        :param iterations:
        :return:
        """
        # if auto_stop and not iterations:
        #     i = 0
        #     while True:
        #         i += 1
        #         old_f = self.forward_probability(observations)
        #         self.do_pass(observations, not (i % 100))
        #         if not (i % 100):
        #             print 'ITERATION #{}'.format(i)
        #             print 'Log likelihood = {}'.format(old_f)
        #             print 'Delta llk = {}'.format(delta_p)
        #         if auto_stop:
        #             delta_p = self.forward_probability(observations) - old_f
        #             if delta_p <= CONVERGENCE_DELTA_LOG_LIKELIHOOD:
        #                 break
        # else:
        for i in range(iterations + 1):
            print 'ITERATION #{}'.format(i)
            old_f = self.forward_probability(observations)
            print 'Log likelihood = {}'.format(old_f)

            self.do_pass(observations, verbose)
            new_f = self.forward_probability(observations)
            delta_p = new_f - old_f
            print "Is this an improvement? {}".format("Yes" if delta_p > 0 else "No")
            print 'New log likelihood = {}'.format(new_f)
            print 'Delta llk = {}'.format(delta_p)
            if auto_stop:
                # if not (i % 100):
                if delta_p <= CONVERGENCE_DELTA_LOG_LIKELIHOOD:
                    break

        print "Finished training at {} iterations.".format(i)
        print "DELTA FORWARD LOG PROB AFTER TRAINING:", delta_p
        print "{} --> {}".format(old_f, new_f)
        # print delta_p / np.exp(old_f)

    def emit(self):
        obs = self.emissionDistributions[self.current_state].emit()
        # print "Emitted {}".format(obs)
        return obs

    def transition(self):
        text = "Transition to"
        if self.current_state is not None:
            self.current_state = choice(range(self.nStates), p=self.transitionMatrix[self.current_state,])
        else:
            text = "Start at"
            self.current_state = choice(range(self.nStates), p=self.initialProbabilities)
            # print "{} state{}".format(text, self.current_state)

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
