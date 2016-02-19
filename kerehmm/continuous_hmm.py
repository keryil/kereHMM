from kerehmm.abstract_hmm import AbstractHMM
from kerehmm.distribution import GaussianMixture


class ContinuousHMM(AbstractHMM):
    """
    I am an HMM with continuous Gaussian mixture emission distributions.
    """

    def __init__(self, number_of_states, alphabet_size, state_labels=None, *args, **kwargs):
        super(ContinuousHMM, self).__init__(number_of_states, state_labels, *args, **kwargs)
        self.alphabetSize = alphabet_size
        self.emissionDistributions = [GaussianMixture(n=alphabet_size) for _ in range(self.nStates)]
