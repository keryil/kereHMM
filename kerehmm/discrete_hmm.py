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
            sum = np.logaddexp.reduce(dist.probabilities)
            assert np.isclose(sum, np.log(1))

    def train(self, observations, iterations=5):
        """
        This is the training algorithm in Rabiner 1989
        equations 40a, 40b and 40c.
        :param observations:
        :param iterations:
        :return:
        """

        def do_pass():
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
            print text.format(observations,
                              np.exp(self.initialProbabilities), np.sum(np.exp(self.initialProbabilities)),
                              np.exp(self.transitionMatrix), np.sum(np.exp(self.transitionMatrix), axis=1),
                              emit, np.sum(emit, axis=1))

            # initial probabilities
            pi_new = np.zeros_like(self.initialProbabilities)
            # xi = np.zeros(shape=(len(observations), self.nStates, self.nStates))
            alpha = self.forward(observations)
            beta = self.backward(observations)
            xi = self.xi(observations=observations, alpha=alpha, beta=beta)
            # gamma = np.zeros(shape=(len(observations), self.nStates))  # self.gamma2(observations=observations)
            gamma = self.gamma(xi, observations)
            text = \
                """
                Xi      = {}
                Gamma   = {}
                Alpha   = {}
                Beta    = {}
                """.format(*map(np.exp, [xi, gamma, alpha, beta]))
            print text
            T = len(observations)

            trans_new = np.empty_like(self.transitionMatrix)
            trans_new[:] = -np.inf
            # print np.logaddexp.reduce(np.sum([alpha, beta], axis=0))
            # print np.repeat(np.logaddexp.reduce(np.sum([alpha, beta], axis=1)), alpha.shape[1])
            # gamma[:] = np.sum([alpha, beta], axis=0) - np.expand_dims(np.logaddexp.reduce(np.sum([alpha, beta], axis=0)), axis=0)
            # for t, obs in enumerate(observations):
            #     text = "Alpha[t]: {}\n Beta[t]: {}\n Alpha[t]*Beta[t]: {}".format(*map(np.exp,
            #                                                              [alpha[t], beta[t], alpha[t] + beta[t]]))
            #     text = text.replace("[t]", "[{}]").format(t, t, t, t)
            #     print text
            #     print "sum(Alpha[{}]*Beta[{}]) = {}".format(t, t,np.exp(np.logaddexp.reduce(alpha[t] + beta[t])))
            #     # gamma[t] = alpha[t] + beta[t] - np.logaddexp.reduce(alpha[t] + beta[t])
            #     print "gamma[{}] = {}".format(t, np.exp(gamma[t]))

            pi_new[:] = gamma[0, :]
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
            local_xi = xi[:T - 1]
            local_gamma = gamma[:T - 1]
            for from_, to in product(range(self.nStates), range(self.nStates)):
                # print "State{} --> State{}".format(from_, to)
                sum_xi = np.logaddexp.reduce(local_xi[:, from_, to], axis=0)
                sum_gamma = np.logaddexp.reduce(local_gamma[:, from_], axis=0)

                if not np.isclose(np.exp(self.transitionMatrix[from_, to]), np.exp(sum_xi - sum_gamma)):
                    print "SumXI {} / SumGAMMA {} = {}".format(*map(np.exp, (sum_xi, sum_gamma, sum_xi - sum_gamma)))
                    print "old prob -> new prob: {} -> {}".format(np.exp(self.transitionMatrix[from_, to]),
                                                                  np.exp(sum_xi - sum_gamma))
                trans_new[from_, to] = sum_xi - sum_gamma

            for i, row in enumerate(trans_new):
                print "Row sum:", np.exp(np.logaddexp.reduce(row))

            # print "New transition matrix:", np.exp(trans_new)
            self.transitionMatrix = trans_new
            self.initialProbabilities = pi_new
            self.emissionDistributions = new_dists

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
            print text.format(observations,
                              np.exp(self.initialProbabilities), sum(np.exp(self.initialProbabilities)),
                              np.exp(self.transitionMatrix), np.sum(np.exp(self.transitionMatrix), axis=1),
                              dists, np.sum(dists, axis=1))

        for i in range(iterations):
            do_pass()
