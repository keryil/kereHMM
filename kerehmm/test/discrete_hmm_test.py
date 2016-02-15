import numpy as np

from kerehmm.discrete_hmm import DiscreteHMM


class DiscreteHMMTest(object):
    nStates = 3
    nSymbols = 3

    def new_hmm(self):
        hmm = DiscreteHMM(self.nStates, self.nSymbols)  # , verbose=True)
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

    def test_viterbi_path_with_emissions(self):
        hmm = self.new_hmm()
        hmm.setup_strict_left_to_right(set_emissions=True)

        # now, the true path should always be the same regardless of the input,
        # as long as it is of size nStates
        true_path = np.array(range(self.nStates))

        # random observations
        observations = np.array(range(self.nStates))

        viterbi_path, viterbi_prob = hmm.viterbi_path(observations)
        assert np.array_equal(viterbi_path, true_path)

    def test_viterbi_path_with_transitions(self):
        hmm = self.new_hmm()
        hmm.setup_strict_left_to_right()

        # now, the true path should always be the same regardless of the input,
        # as long as it is of size nStates
        true_path = np.array(range(self.nStates))

        # random observations
        observations = np.random.randint(0, self.nStates - 1, self.nStates)

        viterbi_path, viterbi_prob = hmm.viterbi_path(observations)
        assert np.array_equal(viterbi_path, true_path)

    def test_training(self):
        hmm = self.new_hmm()
        hmm.setup_strict_left_to_right()
        observations = [0, 1, 1, 2, 0]  # ,0,1,2,0,1,2]

        hmm.train(observations, iterations=1)

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
        seq = ghmm.EmissionSequence(hmm_reference.emissionDomain, [0, 0, 0, 0])
        forward = hmm.forward([0, 0, 0, 0])

        # remember that we have to convert stuff from ghmm to log scale
        forward_reference, scale_reference = map(np.array, hmm_reference.forward(seq))
        forward_reference_log = np.log(forward_reference)
        print "Forward reference (scaled):\n", forward_reference

        for i, c in enumerate(scale_reference):
            forward_reference_log[i] += sum(np.log(scale_reference[:i + 1]))
        print "Forward reference (unscaled):\n", np.exp(forward_reference_log)

        print "Forward:\n", np.exp(forward)

        assert np.allclose(forward, forward_reference_log)

    def test_backward_against_ghmm(self):
        from kerehmm.test.util import ghmm_from_discrete_hmm
        import ghmm
        hmm = self.new_hmm()
        hmm_reference = ghmm_from_discrete_hmm(hmm)
        seq = ghmm.EmissionSequence(hmm_reference.emissionDomain, [0, 0, 0])

        # remember that we have to convert stuff from ghmm to log scale
        _, scale_reference = hmm_reference.forward(seq)
        # print "Forward referece", forward
        print "Scale reference", scale_reference

        # this is the reference backward array, untransformed (scaled)
        backward_reference = np.array(hmm_reference.backward(seq, scalingVector=scale_reference))
        print "Backward reference (scaled)", backward_reference

        # unscale the reference array
        # get the product of scale_t,scale_t+1,...,scale_T for each t.
        coefficients = np.array([np.prod(scale_reference[i:]) for i, _ in enumerate(scale_reference)])
        print "Reference coefficients:", coefficients

        # multiply each backwards_reference[i] by coefficients[i]
        backward_reference[:] = np.multiply(backward_reference, coefficients).T

        # test shape
        print "Backward reference (unscaled)", backward_reference

        # this is our backward array, log transformed
        backward = hmm.backward([0, 0, 0])

        print "Backward", np.exp(backward)

        assert backward.shape == backward_reference.shape

        # test values
        # print "Diff:", np.exp(backward) - backward_reference
        backward_unscaled = np.exp(backward)
        assert np.allclose(backward_unscaled, backward_reference)

    def test_viterbi_against_hmm(self):
        from kerehmm.test.util import ghmm_from_discrete_hmm
        import ghmm

        hmm = self.new_hmm()
        hmm.setup_strict_left_to_right(set_emissions=True)
        domain = ghmm.Alphabet(range(hmm.alphabetSize))

        hmm_reference = ghmm_from_discrete_hmm(hmm)
        seq = list(range(self.nSymbols))
        print "True path and emission: {}".format(seq)
        true_path = seq
        reference_path, reference_prob = hmm_reference.viterbi(ghmm.SequenceSet(domain,
                                                                                [seq]))
        path, prob = hmm.viterbi_path(seq)
        print "Reference path: {}".format(reference_path)
        print "Calculated path: {}".format(path)
        print "Reference prob: {}, Calculated prob: {}".format(reference_prob, prob)
        assert np.all(np.equal(true_path, reference_path))
        assert np.all(np.equal(true_path, path))
        assert np.isclose(prob, reference_prob)
