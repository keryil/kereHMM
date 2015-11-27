from abstract_hmm import AbstractHMM
from distribution import DiscreteDistribution


class DiscreteHMM(AbstractHMM):
    """
    I am an HMM with discrete emission distributions.
    """

    def __init__(self, number_of_states, state_labels=None):
        super(DiscreteHMM, AbstractHMM).__init__(number_of_states, state_labels)
        self.emissionDistributions = [DiscreteDistribution() for _ in range(self.nStates)]

    def observation_probability(self, observations):
        """
        This implements
        :param observations:
        :return:
        """
        pass
