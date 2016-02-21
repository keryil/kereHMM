from copy import deepcopy
from itertools import product

import numpy as np

from kerehmm.util import DELTA_P
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
        emit = map(lambda x: np.exp(x.probabilities), self.emissionDistributions)
        if verbose:
            print text.format(observations,
                              np.exp(self.initialProbabilities), np.sum(np.exp(self.initialProbabilities)),
                              np.exp(self.transitionMatrix), np.sum(np.exp(self.transitionMatrix), axis=1),
                              emit, np.sum(emit, axis=1))
        # print "FORWARD PROB: {}".format(np.exp(self.forward_probability(observations)))
        # initial probabilities
        pi_new = np.zeros_like(self.initialProbabilities)
        # xi = np.zeros(shape=(len(observations), self.nStates, self.nStates))
        alpha = self.forward(observations)
        beta = self.backward(observations)
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

        pi_new[:] = gamma[0, :]
        if verbose:
            print "New initial probs:", np.exp(pi_new)

        new_dists = deepcopy(self.emissionDistributions)
        for state, dist in enumerate(new_dists):
            denominator = np.logaddexp.reduce(gamma[:, state])
            for symbol in range(self.alphabetSize):
                nominator = gamma[np.array(observations) == symbol, state]
                if len(nominator):
                    nominator = np.logaddexp.reduce(gamma[np.array(observations) == symbol, state])
                else:
                    nominator = np.log(DELTA_P)
                    denominator = np.logaddexp(np.log(DELTA_P), denominator)
                dist.probabilities[symbol] = nominator
            dist.probabilities -= denominator
        # print "New emission distributions: {}".format(map(lambda x: np.exp(x.probabilities), new_dists))

        # transition matrix
        # TODO: there might be a problem with the backwards probabilities.
        trans_new = np.empty_like(self.transitionMatrix)
        # trans_new[:] = -np.inf
        # local_xi = xi[:T - 1]
        # local_gamma = gamma[:T - 1]
        for from_, to in product(range(self.nStates), range(self.nStates)):
            sum_xi = -np.inf
            sum_gamma = -np.inf
            for t in range(T - 1):
                # print "State{} --> State{}".format(from_, to)
                sum_xi = np.logaddexp(sum_xi, xi[t, from_, to])
                sum_gamma = np.logaddexp(sum_gamma, gamma[t, from_])
            trans_new[from_, to] = sum_xi - sum_gamma

        for i, row in enumerate(trans_new):
            if verbose:
                print "Row sum:", np.exp(np.logaddexp.reduce(row))

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
        dists = map(lambda x: np.exp(x.probabilities), self.emissionDistributions)
        if verbose:
            print text.format(observations,
                              np.exp(self.initialProbabilities), sum(np.exp(self.initialProbabilities)),
                              np.exp(self.transitionMatrix), np.sum(np.exp(self.transitionMatrix), axis=1),
                              dists, np.sum(dists, axis=1))