import numpy as np

from kerehmm.discrete_hmm import DiscreteHMM
from kerehmm.util import random_simplex


class DiscreteHMMTest(object):
    nStates = 3
    nSymbols = 3

    def new_hmm(self, random_transitions=False, random_emissions=False):
        hmm = DiscreteHMM(self.nStates, self.nSymbols, random_transitions=random_transitions,
                          random_emissions=random_emissions)  # , verbose=True)
        return hmm

    def to_ghmm(self, hmm):
        from .util import ghmm_from_discrete_hmm
        return ghmm_from_discrete_hmm(hmm)


class TestGhmmConversion(DiscreteHMMTest):
    def test_probabilities(self):
        hmm = self.new_hmm()
        hmm_reference = self.to_ghmm(hmm)

        trans = hmm.transitionMatrix
        emit = [d.probabilities for d in hmm.emissionDistributions]
        init = hmm.initialProbabilities

        trans_reference, emit_reference, init_reference = hmm_reference.asMatrices()
        assert np.array_equal(trans, trans_reference)
        assert np.array_equal(init, init_reference)
        assert np.array_equal(emit, emit_reference)


class TestStandalone(DiscreteHMMTest):
    def test_random_init(self):
        for i in range(1000):
            self.new_hmm(random_transitions=True, random_emissions=True)

    def test_array_sizes(self):
        hmm = self.new_hmm()
        assert len(hmm.emissionDistributions) == self.nStates
        assert hmm.transitionMatrix.shape == (self.nStates, self.nStates)
        assert hmm.initialProbabilities.shape == (self.nStates,)
        assert all([d.n == self.nSymbols for d in hmm.emissionDistributions])

    def test_forward_probabilities(self):
        hmm = self.new_hmm()
        prob_trans = 1. / self.nStates
        prob_emit = 1. / self.nSymbols
        hmm.transitionMatrix[:] = prob_trans
        hmm.initialProbabilities[:] = prob_trans
        for emission in hmm.emissionDistributions:
            emission.probabilities[:] = prob_emit

        forward, coefs = hmm.forward([0, 0, 0])
        prob_line = np.array([(prob_trans * prob_emit)] * self.nStates)

        print "Forward: {}".format(forward)
        print "Scale: {}".format(coefs)
        # these should be equivalent under such parameters
        assert np.array_equal(forward, hmm.forward([1, 1, 1])[0])
        assert np.array_equal(forward[0] * coefs[0], prob_line)

        prob_line[:] = (prob_line * prob_trans).sum() * prob_emit
        assert np.array_equal(forward[1] * np.product(coefs[:2]), prob_line)

        prob_line[:] = (prob_line * prob_trans).sum() * prob_emit
        assert np.array_equal(forward[2] * np.product(coefs[:3]), prob_line)

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

    def test_training_from_simulation(self):
        self.nStates = 3
        self.nSymbols = 3
        observation_size = 300

        hmm = self.new_hmm(random_transitions=True, random_emissions=True)
        # hmm.setup_strict_left_to_right()
        reference_hmm = self.new_hmm(random_emissions=True)  # , random_transitions=True)
        reference_hmm.setup_left_to_right()
        # reference_hmm.current_state = 0
        true_init_p = reference_hmm.initialProbabilities
        true_trans_p = reference_hmm.transitionMatrix
        true_emission_p = np.array(map(lambda x: x.probabilities, reference_hmm.emissionDistributions))
        true_states, observations = reference_hmm.simulate(iterations=observation_size)

        text = \
            """
            True init probs:
            {}
            Diff:
            {}: {}
            True emission probs:
            {}
            Diff:
            {}: {}
            True trans probs:
            {}
            Diff:
            {}: {}
            Observations ({}):
            {}
                        """
        print text.format(true_init_p, hmm.initialProbabilities,
                          np.sum(np.abs(true_init_p - hmm.initialProbabilities)),
                          true_emission_p, np.array([p.probabilities for p in hmm.emissionDistributions]),
                          np.sum(np.abs([p1 - p2.probabilities for p1, p2 in
                                         zip(true_emission_p, hmm.emissionDistributions)])),
                          true_trans_p, hmm.transitionMatrix,
                          np.sum(np.abs(true_trans_p - hmm.transitionMatrix)),
                          observation_size, observations)

        hmm.train(observations, auto_stop=False, iterations=10)
        print text.format(true_init_p, hmm.initialProbabilities,
                          np.sum(np.abs(true_init_p - hmm.initialProbabilities)),
                          true_emission_p, np.array([p.probabilities for p in hmm.emissionDistributions]),
                          np.sum(np.abs([p1 - p2.probabilities for p1, p2 in
                                         zip(true_emission_p, hmm.emissionDistributions)])),
                          true_trans_p, hmm.transitionMatrix,
                          np.sum(np.abs(true_trans_p - hmm.transitionMatrix)),
                          observation_size, observations)

    def test_training(self):
        self.nStates = 3
        self.nSymbols = 2
        from numpy.random import choice
        observation_size = 1000

        hmm = self.new_hmm(random_transitions=True, random_emissions=True)
        # hmm.setup_strict_left_to_right()
        true_init_p = random_simplex(self.nStates)
        true_states = [choice(range(self.nStates), p=true_init_p)]
        true_trans_p = np.array([[.50, .15, .35],
                                 [.10, .80, .10],
                                 [.20, .10, .70]])  # random_simplex(self.nStates, two_d=True)
        for i in range(1, observation_size):
            true_states.append(choice(range(self.nStates), p=true_trans_p[true_states[-1]]))
        true_emission_p = np.array([[.70, .30],
                                    [.40, .60],
                                    [.10, .90]])  # [random_simplex(self.nSymbols) for _ in range(self.nStates)]
        observations = [choice(range(self.nSymbols), p=true_emission_p[state]) for state in true_states]

        text = \
            """
            True init probs:
            {}
            Ours:
            {}
            True emission probs:
            {}
            Ours:
            {}
            True trans probs:
            {}
            Ours:
            {}
            Observations ({}):
            {}
                        """
        print text.format(true_init_p, hmm.initialProbabilities,
                          true_emission_p, np.array([p.probabilities for p in hmm.emissionDistributions]),
                          true_trans_p, hmm.transitionMatrix,
                          observation_size, observations)

        hmm.train(observations, auto_stop=True, iterations=30)
        print text.format(true_init_p, hmm.initialProbabilities,
                          true_emission_p, np.array([p.probabilities for p in hmm.emissionDistributions]),
                          true_trans_p, hmm.transitionMatrix,
                          observation_size, observations)


class TestAgainstGhmm(DiscreteHMMTest):

    def test_forward_against_ghmm(self):
        from .util import ghmm_from_discrete_hmm
        import ghmm
        hmm = self.new_hmm(random_transitions=True, random_emissions=True)
        hmm_reference = ghmm_from_discrete_hmm(hmm)
        observation_size = 10
        observed = np.random.choice(range(self.nSymbols), size=observation_size).tolist()
        seq = ghmm.EmissionSequence(hmm_reference.emissionDomain, observed)
        forward, scale = hmm.forward(observed)

        # remember that we have to convert stuff from ghmm to log scale
        forward_reference, scale_reference = map(np.array, hmm_reference.forward(seq))
        print "Forward reference (scaled):\n", forward_reference
        print "Scale reference: {}".format(scale_reference)
        # for i, c in enumerate(scale_reference):
        #     forward_reference_log[i] += sum(np.log(scale_reference[:i + 1]))
        # print "Forward reference (unscaled):\n", np.exp(forward_reference_log)

        print "Forward:\n", forward
        print "Scale:\n", scale

        assert np.allclose(forward, forward_reference)
        assert np.allclose(scale, scale_reference)

    def test_backward_against_ghmm(self):
        from kerehmm.test.util import ghmm_from_discrete_hmm
        import ghmm
        hmm = self.new_hmm(random_emissions=True, random_transitions=True)
        hmm_reference = ghmm_from_discrete_hmm(hmm)
        observation_size = 10
        observed = np.random.choice(range(self.nSymbols), size=observation_size).tolist()
        seq = ghmm.EmissionSequence(hmm_reference.emissionDomain, observed)
        _, scale = hmm.forward(observations=observed)
        # remember that we have to convert stuff from ghmm to log scale
        _, scale_reference = map(np.array, hmm_reference.forward(seq))
        # print "Forward referece", forward
        print "Scale reference", scale_reference
        assert np.allclose(scale, scale_reference)

        # this is the reference backward array, untransformed (scaled)
        backward_reference = np.array(hmm_reference.backward(seq, scalingVector=scale_reference))
        print "Backward reference (scaled)", backward_reference

        backward = hmm.backward(observed, scale_coefficients=scale)

        print "Backward", backward

        assert backward.shape == backward_reference.shape

        # test values
        # print "Diff:", np.exp(backward) - backward_reference
        # backward_unscaled = np.exp(backward)
        assert np.allclose(backward, backward_reference)

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
