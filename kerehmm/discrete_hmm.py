from itertools import product

import numpy as np

from .abstract_hmm import AbstractHMM
from .distribution import DiscreteDistribution

add_logs = lambda lst: reduce(np.logaddexp, lst)

class DiscreteHMM(AbstractHMM):
    """
    I am an HMM with discrete emission distributions.
    """

    def __init__(self, number_of_states, alphabet_size, state_labels=None, *args, **kwargs):
        super(DiscreteHMM, self).__init__(number_of_states, state_labels, *args, **kwargs)
        self.alphabetSize = alphabet_size
        self.emissionDistributions = [DiscreteDistribution(n=alphabet_size) for _ in range(self.nStates)]

    def train(self, observations, iterations=5):
        """
        This is the training algorithm in Rabiner 1989
        equations 40a, 40b and 40c.
        :param observations:
        :param iterations:
        :return:
        """

        def do_pass():
            # initial probabilities
            pi_new = np.zeros_like(self.initialProbabilities)
            xi = self.xi(observations=observations)
            gamma = np.zeros(shape=(len(observations), self.nStates))  # self.gamma2(observations=observations)
            alpha = self.forward(observations)
            beta = self.backward(observations)
            T = len(observations)

            trans_new = np.empty_like(self.transitionMatrix)
            trans_new[:] = -np.inf

            for t, o in enumerate(observations):
                gamma[t] = alpha[t] + beta[t] - np.logaddexp.reduce(alpha[t] + beta[t])

                # handle the transition matrix
                if t == 0:
                    pi_new[:] = gamma[t]
                    print "New initial probs:", np.exp(pi_new)
                    assert np.isclose(np.logaddexp.reduce(pi_new), 0.)

            # transition matrix
            for state1, state2 in product(range(self.nStates), range(self.nStates)):
                local_xi = xi[:T - 1, state1, state2]
                local_gamma = gamma[:T - 1, state1]
                sum_xi = np.logaddexp.reduce(local_xi)
                sum_gamma = np.logaddexp.reduce(local_gamma)
                print 'xi', np.exp(local_xi), np.exp(sum_xi)
                print 'gamma', np.exp(local_gamma), np.exp(sum_gamma)
                print "new prob", np.exp(sum_xi - sum_gamma)
                trans_new[state1, state2] = np.exp(sum_xi - sum_gamma)

            for i, _ in enumerate(trans_new):
                print "Row sum:", np.exp(np.logaddexp.reduce(trans_new[i]))

            print "New transition matrix:", np.exp(trans_new)
            # print "Sums", np.sum.reduce(np.exp(trans_new), axis=1)
            # # emission probabilities
            # denominator = np.logaddexp.reduce(gamma)
            # for state, distribution in enumerate(self.emissionDistributions):
            #     state_map = np.array([np.log(1) if o == state else -np.inf for o in observations])
            #     # print "Map:", np.atleast_2d(state_map).T
            #     mapped_gamma = gamma + np.atleast_2d(state_map).T
            #     # print "Mapped gamma:", mapped_gamma

            # def do_pass():
            #     # initial probabilities
            #     pi_new = np.zeros_like(self.initialProbabilities)
            #     xi = self.xi(observations=observations)
            #     gamma = self.gamma2(observations=observations)
            #     pi_new[:] = gamma[0]
            #     print "New initial probs:", np.exp(pi_new)
            #
            #     # transition matrix
            #     trans_new = np.empty_like(self.transitionMatrix)
            #     trans_new[:] = -np.inf
            #     for i, _ in enumerate(self.transitionMatrix):
            #         for j, _ in enumerate(self.transitionMatrix[i]):
            #             T = len(observations)
            #             trans_new[i, j] = np.logaddexp.reduce(xi[:T - 1, i, j], axis=0) - \
            #                               np.logaddexp.reduce(gamma[:T - 1, i], axis=0)

            # print "New transition matrix:", np.exp(trans_new)
            # emission probabilities
            # denominator = np.logaddexp.reduce(gamma)
            # for state, distribution in enumerate(self.emissionDistributions):
            #     state_map = np.array([np.log(1) if o == state else -np.inf for o in observations])
            #     # print "Map:", np.atleast_2d(state_map).T
            #     mapped_gamma = gamma + np.atleast_2d(state_map).T
            #     # print "Mapped gamma:", mapped_gamma

        for i in range(iterations):
            do_pass()
