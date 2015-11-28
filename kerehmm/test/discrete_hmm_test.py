import pytest
from kerehmm.discrete_hmm import DiscreteHMM
import numpy as np


class TestDiscreteHMM(object):
    nStates = 4
    nSymbols = 5

    def new_hmm(self):
        hmm = DiscreteHMM(self.nStates, self.nSymbols)
        return hmm

    def test_array_sizes(self):
        hmm = self.new_hmm()
        assert len(hmm.emissionDistributions) == self.nStates
        assert hmm.transitionMatrix.shape == (self.nStates, self.nStates)
        assert hmm.initialProbabilities.shape == (self.nStates,)
        assert all([d.n == self.nSymbols for d in hmm.emissionDistributions])

    def test_forward_probabilities(self):
        hmm = self.new_hmm()
        prob_trans = np.log(1. / self.nStates)
        prob_emit = np.log(1. / self.nSymbols)
        hmm.transitionMatrix[:] = prob_trans
        hmm.initialProbabilities[:] = prob_trans
        for emission in hmm.emissionDistributions:
            emission.probabilities[:] = prob_emit
        prob_line = np.array([(prob_trans + prob_emit)] * self.nStates)

        assert np.array_equal(hmm.forward([0, 0, 0]), hmm.forward([1, 1, 1]))

        assert np.array_equal(hmm.forward([0, 0, 0])[0], prob_line)
        prob_line[:] = np.logaddexp.reduce(prob_line + prob_trans) + prob_emit
        assert np.array_equal(hmm.forward([0, 0, 0])[1], prob_line)
        prob_line[:] = np.logaddexp.reduce(prob_line + prob_trans) + prob_emit
        assert np.array_equal(hmm.forward([0, 0, 0])[2], prob_line)

    def test_viterbi_path(self):
        hmm = self.new_hmm()
        # zero out all transitions but one
        delta_p = .00001
        hmm.transitionMatrix[:] = np.log(delta_p)
        for state in range(self.nStates - 1):
            hmm.transitionMatrix[state, state + 1] = np.log(1 - delta_p * (self.nStates - 1))

        # set the initial probs the same way
        hmm.initialProbabilities = np.array([np.log(1 - delta_p * (self.nStates - 1))] + \
                                            [delta_p] * (self.nStates - 1))

        # now, the true path should always be the same regardless of the input,
        # as long as it is of size nStates
        true_path = np.array(range(self.nStates))

        # random observations
        observations = np.random.randint(0, self.nStates - 1, self.nStates)

        viterbi_path, viterbi_prob = hmm.viterbi_path(observations)
        assert np.array_equal(viterbi_path, true_path)
