import ghmm
import numpy as np


def ghmm_from_discrete_hmm(hmm):
    domain = ghmm.Alphabet(range(hmm.alphabetSize))
    trans = np.exp(hmm.transitionMatrix)
    init = np.exp(hmm.initialProbabilities)
    emissions = np.exp([d.probabilities for d in hmm.emissionDistributions])
    return ghmm.HMMFromMatrices(emissionDomain=domain,
                                distribution=ghmm.DiscreteDistribution(domain),
                                A=trans,
                                B=emissions,
                                pi=init)


if __name__ == "__main__":
    from discrete_hmm_test import DiscreteHMMTest

    test = DiscreteHMMTest()
    hmm = test.new_hmm()
    domain = ghmm.Alphabet(range(hmm.alphabetSize))
    hmm_reference = ghmm_from_discrete_hmm(hmm)
    seq = ghmm.EmissionSequence(hmm_reference.emissionDomain, [0, 0, 0])
    print hmm_reference
    print hmm_reference.forward(seq)[0]
    print hmm_reference.backward(seq)[0]
