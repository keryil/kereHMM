from copy import deepcopy
from itertools import product

import numpy as np

from kerehmm.util import DELTA_P, smooth_probabilities
from .abstract_hmm import AbstractHMM
from .distribution import DiscreteDistribution


# add_logs = lambda lst: reduce(np.logaddexp, lst)

class DiscreteHMM(AbstractHMM):
    """
    I am an HMM with discrete emission distributions.
    """

    def __init__(self, number_of_states, alphabet_size, state_labels=None, random_emissions=False, *args, **kwargs):
        super(DiscreteHMM, self).__init__(number_of_states, state_labels, *args, **kwargs)
        self.alphabetSize = alphabet_size
        self.emissionDistributions = [DiscreteDistribution(n=alphabet_size, randomize=random_emissions) for _ in
                                      range(self.nStates)]
        self.sanity_check()

    def sanity_check(self):
        super(DiscreteHMM, self).sanity_check()
        for dist in self.emissionDistributions:
            assert np.isclose(dist.probabilities.sum(), 1.)

    def do_pass(self, observations, verbose=False):
        text = \
            """
            --------------------------<>
            do_pass() called
                observation: {}
            Parameters before iteration:
                initial: {} = {}
                transition: {} = {}
                emission: {} = {}
            --------------------------<>
            """
        emit = map(lambda x: x.probabilities, self.emissionDistributions)
        if verbose:
            print text.format(observations,
                              self.initialProbabilities, np.sum(self.initialProbabilities),
                              self.transitionMatrix, np.sum(self.transitionMatrix, axis=1),
                              emit, np.sum(emit, axis=1))
        # print "FORWARD PROB: {}".format(np.exp(self.forward_probability(observations)))
        # initial probabilities
        pi_new = np.zeros_like(self.initialProbabilities)
        # xi = np.zeros(shape=(len(observations), self.nStates, self.nStates))
        alpha, scale = self.forward(observations)
        beta = self.backward(observations, scale_coefficients=scale)
        xi = self.xi(observations=observations, alpha=alpha, beta=beta)
        # gamma = np.zeros(shape=(len(observations), self.nStates))  # self.gamma2(observations=observations)
        gamma = self.gamma(alpha, beta)
        # text = \
        #     """
        #     Xi      = {}
        #     Gamma   = {}
        #     Alpha   = {}
        #     Beta    = {}
        #     """.format(*map(np.exp, [xi, gamma, alpha, beta]))
        # print text
        T = len(observations)

        pi_new[:] = smooth_probabilities(gamma[0, :])
        if verbose:
            print "New initial probs:", pi_new

        new_dists = deepcopy(self.emissionDistributions)
        for state, dist in enumerate(new_dists):
            denominator = gamma[:, state].sum()
            for symbol in range(self.alphabetSize):
                nominator = gamma[np.array(observations) == symbol, state]
                if len(nominator):
                    nominator = (gamma[np.array(observations) == symbol, state].sum())
                else:
                    nominator = DELTA_P
                    denominator += DELTA_P
                dist.probabilities[symbol] = nominator
            dist.probabilities /= denominator
            dist.probabilities = smooth_probabilities(dist.probabilities)
        # print "New emission distributions: {}".format(map(lambda x: np.exp(x.probabilities), new_dists))

        # transition matrix
        # TODO: there might be a problem with the backwards probabilities.
        trans_new = np.empty_like(self.transitionMatrix)
        # trans_new[:] = -np.inf
        # local_xi = xi[:T - 1]
        # local_gamma = gamma[:T - 1]
        for from_, to in product(range(self.nStates), range(self.nStates)):
            sum_xi = xi[:T - 1, from_, to].sum()
            sum_gamma = gamma[:T - 1, from_].sum()
            trans_new[from_, to] = sum_xi / sum_gamma
        trans_new /= np.expand_dims(trans_new.sum(axis=1), axis=-1)
        trans_new = smooth_probabilities(trans_new)

        for i, row in enumerate(trans_new):
            if verbose:
                print "Row sum:", row.sum()

        # print "New transition matrix:", np.exp(trans_new)
        self.initialProbabilities = pi_new
        self.sanity_check()
        self.emissionDistributions = new_dists
        self.sanity_check()
        self.transitionMatrix = trans_new
        self.sanity_check()

        text = \
            """
            --------------------------<>
            do_pass() returning
                observation: {}
            Parameters after iteration:
                initial: {} = {}
                transition: {} = {}
                emission: {} = {}
            --------------------------<>
            """
        dists = map(lambda x: x.probabilities, self.emissionDistributions)
        if verbose:
            print text.format(observations,
                              self.initialProbabilities, self.initialProbabilities.sum(),
                              self.transitionMatrix, self.transitionMatrix.sum(axis=1),
                              dists, np.sum(dists, axis=1))

    def initialize_parameters(self, observations):
        pass
