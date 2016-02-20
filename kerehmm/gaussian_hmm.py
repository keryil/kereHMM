from copy import deepcopy
from itertools import product

import numpy as np

from kerehmm.abstract_hmm import AbstractHMM
from kerehmm.distribution import GaussianDistribution


class GaussianHMM(AbstractHMM):
    """
    I am an HMM with continuous Gaussian emission distributions.
    """

    def __init__(self, number_of_states, dimensions, random_emissions=False, state_labels=None, *args, **kwargs):
        super(GaussianHMM, self).__init__(number_of_states, state_labels, *args, **kwargs)
        if random_emissions:
            if "upper_bounds" not in kwargs:
                kwargs['upper_bounds'] = [100 for _ in range(dimensions)]
            if "lower_bounds" not in kwargs:
                kwargs['lower_bounds'] = [0 for _ in range(dimensions)]
            self.emissionDistributions = [GaussianDistribution(dimensions, random=random_emissions,
                                                               upper_bounds=kwargs['upper_bounds'],
                                                               lower_bounds=kwargs['lower_bounds'])
                                          for _ in range(self.nStates)]
        else:
            self.emissionDistributions = [GaussianDistribution(dimensions) for _ in range(self.nStates)]

    def do_pass(self, observations, verbose=False):
        text = \
            """
            --------------------------<>
            do_pass() called
                observation: {}
            Parameters before iteration:
                initial: {} = {}
                transition: {} = {}
                emission: {}
            --------------------------<>
            """
        emit = self.emissionDistributions
        if verbose:
            print text.format(observations,
                              np.exp(self.initialProbabilities), np.sum(np.exp(self.initialProbabilities)),
                              np.exp(self.transitionMatrix), np.sum(np.exp(self.transitionMatrix), axis=1),
                              emit)
        # print "FORWARD PROB: {}".format(np.exp(self.forward_probability(observations)))
        # initial probabilities
        pi_new = np.zeros_like(self.initialProbabilities)
        # xi = np.zeros(shape=(len(observations), self.nStates, self.nStates))
        alpha = self.forward(observations)
        beta = self.backward(observations)
        xi = self.xi(observations=observations, alpha=alpha, beta=beta)
        # gamma = np.zeros(shape=(len(observations), self.nStates))  # self.gamma2(observations=observations)
        gamma = self.gamma3(alpha, beta)
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
            nominator = np.exp(gamma[:, state])
            for t, o in enumerate(observations):
                nominator[t] *= o
            # try:
            #     len(new_dists[state].mean)
            new_dists[state].mean = nominator.sum(axis=0) / np.exp(denominator)
            # except TypeError:
            #     new_dists[state].mean = np.exp(np.logaddexp.reduce(nominator, axis=0) - denominator)[0]

        print "New emission distributions: {}".format(new_dists)

        # transition matrix
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
        self.sanity_check(verbose=verbose)  # , sanitize=True)

        text = \
            """
            --------------------------<>
            do_pass() returning
                observation: {}
            Parameters after iteration:
                initial: {} = {}
                transition: {} = {}
                emission: {}
            --------------------------<>
            """
        dists = self.emissionDistributions
        if verbose:
            print text.format(observations,
                              np.exp(self.initialProbabilities), sum(np.exp(self.initialProbabilities)),
                              np.exp(self.transitionMatrix), np.sum(np.exp(self.transitionMatrix), axis=1),
                              dists)
