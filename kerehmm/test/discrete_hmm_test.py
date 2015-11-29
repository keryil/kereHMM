import pytest
from kerehmm.discrete_hmm import DiscreteHMM
import numpy as np


class DiscreteHMMTest(object):
    nStates = 4
    nSymbols = 5

    def new_hmm(self):
        hmm = DiscreteHMM(self.nStates, self.nSymbols)
        return hmm

    def to_ghmm(self, hmm):
        from .util import ghmm_from_discrete_hmm
        return ghmm_from_discrete_hmm(hmm)


class TestGhmmConversion(DiscreteHMMTest):
    def test_probabilities(self):
        hmm = self.new_hmm()
        hmm_reference = self.to_ghmm(hmm)
        trans = np.exp(hmm.transitionMatrix)
        emit = np.exp([d.probabilities for d in hmm.emissionDistributions])
        init = np.exp(hmm.initialProbabilities)
        # trans_reference = hmm_reference.A
        trans_reference, emit_reference, init_reference = hmm_reference.asMatrices()
        assert np.array_equal(trans, trans_reference)
        assert np.array_equal(init, init_reference)
        assert np.array_equal(emit, emit_reference)


class TestStandalone(DiscreteHMMTest):
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
        hmm.setup_strict_left_to_right()

        # now, the true path should always be the same regardless of the input,
        # as long as it is of size nStates
        true_path = np.array(range(self.nStates))

        # random observations
        observations = np.random.randint(0, self.nStates - 1, self.nStates)

        viterbi_path, viterbi_prob = hmm.viterbi_path(observations)
        assert np.array_equal(viterbi_path, true_path)

        # def test_gamma(self):
        #     hmm = self.new_hmm()
        #     gamma = hmm.gamma(observations=range(self.nStates))
        #     for row in gamma:
        #         assert np.abs(1 - np.exp(np.logaddexp.reduce(row))) < .0000001


class TestAgainstGhmm(DiscreteHMMTest):
    def test_forward_against_ghmm(self):
        from .util import ghmm_from_discrete_hmm
        import ghmm
        hmm = self.new_hmm()
        hmm_reference = ghmm_from_discrete_hmm(hmm)
        seq = ghmm.EmissionSequence(hmm_reference.emissionDomain, [0, 0, 0])
        forward = np.exp(hmm.forward([0, 0, 0]))
        forward_reference = hmm_reference.forward(seq)[0]
        assert np.array_equal(forward, forward_reference)
