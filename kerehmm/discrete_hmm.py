from .abstract_hmm import AbstractHMM
from .distribution import DiscreteDistribution


class DiscreteHMM(AbstractHMM):
    """
    I am an HMM with discrete emission distributions.
    """

    def __init__(self, number_of_states, alphabet_size, state_labels=None):
        super(DiscreteHMM, self).__init__(number_of_states, state_labels)
        self.alphabetSize = alphabet_size
        self.emissionDistributions = [DiscreteDistribution(n=alphabet_size) for _ in range(self.nStates)]

    def train(self, observations):
        pass
