import pytest
from kerehmm.discrete_hmm import DiscreteHMM


class TestDiscreteHMM(object):
    def new_hmm(self):
        self.nStates = 5
        self.nSymbols = 5
        hmm = DiscreteHMM(self.nStates, self.nSymbols)
        return hmm

    def test_array_sizes(self):
        hmm = self.new_hmm()
        assert len(hmm.emissionDistributions) == self.nStates
        assert hmm.transitionMatrix.shape == (self.nStates, self.nStates)
        assert all([d.n == self.nSymbols for d in hmm.emissionDistributions])
