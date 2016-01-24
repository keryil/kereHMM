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
            gamma = self.gamma2(observations=observations)
            pi_new[:] = gamma[0]
            print "New initial probs:", np.exp(pi_new)

            # transition matrix
            trans_new = np.empty_like(self.transitionMatrix)
            trans_new[:] = -np.inf
            for i, _ in enumerate(self.transitionMatrix):
                for j, _ in enumerate(self.transitionMatrix[i]):
                    T = len(observations)
                    trans_new[i, j] = np.logaddexp.reduce(xi[:T - 1, i, j], axis=0) - \
                                      np.logaddexp.reduce(gamma[:T - 1, i], axis=0)

            print "New transition matrix:", np.exp(trans_new)
            # emission probabilities
            denominator = np.logaddexp.reduce(gamma)
            for state, distribution in enumerate(self.emissionDistributions):
                state_map = np.array([np.log(1) if o == state else -np.inf for o in observations])
                # print "Map:", np.atleast_2d(state_map).T
                mapped_gamma = gamma + np.atleast_2d(state_map).T
                # print "Mapped gamma:", mapped_gamma

        for i in range(iterations):
            do_pass()
