from copy import deepcopy
from itertools import product

import numpy as np

from kerehmm.abstract_hmm import AbstractHMM
from kerehmm.distribution import DiscreteDistribution
from kerehmm.util import smooth_probabilities, DELTA_P


class DiscreteHMM(AbstractHMM):
    """
    I am an HMM with discrete emission distributions.
    """

    def __init__(self, number_of_states, alphabet_size, state_labels=None, random_emissions=False, *args, **kwargs):
        super(DiscreteHMM, self).__init__(number_of_states, state_labels, *args, **kwargs)
        self.alphabetSize = alphabet_size
        self.emissionDistributions = np.array(
            [DiscreteDistribution(n=alphabet_size, randomize=random_emissions) for _ in
             range(self.nStates)])
        self.sanity_check()

    def sanity_check(self):
        super(DiscreteHMM, self).sanity_check()
        for dist in self.emissionDistributions:
            assert np.isclose(dist.probabilities.sum(), 1.)

    def compute_mapping_to(self, to_hmm):
        """
        Function that tries to align two HMMs by making their most similar
        states coincide. Returns a list of how to map from_hmm states
        onto those of to_hmm.

        Let's have two simple and equivalent HMMs that only differ in
        the state order, and are uniform except initial probs.
        >>> import numpy as np
        >>> hmm1 = DiscreteHMM(3, 3)
        >>> hmm1.initialProbabilities = np.array([.1, .7, .8])
        >>> hmm1.initialProbabilities
        array([ 0.1,  0.7,  0.8])
        >>> hmm2 = DiscreteHMM(3, 3)
        >>> hmm2.initialProbabilities = np.array([.8, .7, .1])
        >>> hmm2.initialProbabilities
        array([ 0.8,  0.7,  0.1])
        >>> hmm1.compute_mapping_to(hmm2)
        (2, 1, 0)

        differences are in transition probs
        swapped states 1&2.
        >>> hmm1 = DiscreteHMM(3, 3)
        >>> hmm1.transitionMatrix = np.array([[.9, 0,.1],\
                                              [.2,.8, 0],\
                                              [0 ,.3,.7]])
        >>> hmm2 = DiscreteHMM(3, 3)
        >>> hmm2.transitionMatrix = np.array([[.9, .1,.0],\
                                              [0 ,.7,.3],\
                                              [.2,.0,.8]])
        >>> hmm1.compute_mapping_to(hmm2)
        (0, 2, 1)

        differences are in emission probs
        swapped states 0&2.
        >>> hmm1 = DiscreteHMM(3, 3)
        >>> hmm1.emissionDistributions[0].probabilities = np.array([.1, .1, .8])
        >>> hmm2 = DiscreteHMM(3, 3)
        >>> hmm2.emissionDistributions[2].probabilities = np.array([.1, .1, .8])

        >>> hmm1.compute_mapping_to(hmm2)
        (2, 1, 0)

        :param to_hmm:
        :return:
        """
        return super(DiscreteHMM, self).compute_mapping_to(to_hmm)

    def compute_emission_mapping_score(self, to_hmm, mapping):
        # print self.emissionDistributions
        # print mapping
        # import sys
        # sys.stdout.flush()
        # sys.stderr.flush()
        from_emissions = self.emissionDistributions[mapping]
        to_emissions = to_hmm.emissionDistributions
        # print from_emissions
        # print to_emissions
        emission_score = 0.
        for fro, to in zip(from_emissions, to_emissions):
            # print "Fro:", fro.probabilities, "To:", to.probabilities
            assert isinstance(fro, DiscreteDistribution)
            # print "Emission score:", fro.b_distance(to)
            emission_score += fro.b_distance(to)
        return emission_score

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
        # zi = np.zeros(shape=(len(observations), self.nStates, self.nStates))
        alpha, scale = self.forward(observations)
        beta = self.backward(observations, scale_coefficients=scale)
        xi = self.zi(observations=observations, alpha=alpha, beta=beta)
        # gamma = np.zeros(shape=(len(observations), self.nStates))  # self.gamma2(observations=observations)
        gamma = self.gamma(alpha, beta, scale)
        # text = \
        #     """
        #     Xi      = {}
        #     Gamma   = {}
        #     Alpha   = {}
        #     Beta    = {}
        #     """.format(*map(np.exp, [zi, gamma, alpha, beta]))
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
                    nominator = nominator.sum()
                else:
                    nominator = 0
                dist.probabilities[symbol] = nominator
            dist.probabilities /= denominator
            dist.probabilities = smooth_probabilities(dist.probabilities)
        # print "New emission distributions: {}".format(map(lambda x: np.exp(x.probabilities), new_dists))

        # transition matrix
        # TODO: there might be a problem with the backwards probabilities.
        trans_new = np.empty_like(self.transitionMatrix)
        # trans_new[:] = -np.inf
        # local_xi = zi[:T - 1]
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

    def setup_left_to_right(self, set_emissions=False):
        """
        Converts this HMM into a strictly left-to-right one.
        :return:
        """
        # if self.nStates != self.alphabetSize:
        #     raise ValueError("")

        delta_p = DELTA_P
        self.transitionMatrix[:] = delta_p
        for state in range(self.nStates):
            try:
                self.transitionMatrix[state, state] = - delta_p * (self.nStates - 2) + .5
                self.transitionMatrix[state, state + 1] = - delta_p * (self.nStates - 2) + .5
            # last state cannot have a valid trans[state+1]
            except IndexError:
                assert state == self.nStates - 1
                self.transitionMatrix[state, 0] = - delta_p * (self.nStates - 2) + .5

        # set the initial probs the same way
        self.initialProbabilities = np.array([1 - delta_p * (self.nStates - 1)] + \
                                             [delta_p] * (self.nStates - 1))

        if set_emissions:
            # set emission probabilities so that each state only emits
            # its own label
            for state in range(self.nStates):
                distribution = self.emissionDistributions[state]
                probs = np.empty_like(distribution.probabilities)
                probs[:] = DELTA_P
                probs[state] = 1 - (self.nStates - 1) * DELTA_P
                distribution.probabilities = probs
