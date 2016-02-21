from copy import deepcopy

import ghmm


def ghmm_from_discrete_hmm(hmm):
    hmm = deepcopy(hmm)
    domain = ghmm.Alphabet(range(hmm.alphabetSize))
    trans = hmm.transitionMatrix
    init = hmm.initialProbabilities
    emissions = [d.probabilities for d in hmm.emissionDistributions]
    return ghmm.HMMFromMatrices(emissionDomain=domain,
                                distribution=ghmm.DiscreteDistribution(domain),
                                A=trans,
                                B=emissions,
                                pi=init)


def ghmm_from_gaussian_hmm(hmm):
    hmm = deepcopy(hmm)
    domain = ghmm.Float()
    trans = hmm.transitionMatrix.tolist()
    init = hmm.initialProbabilities.tolist()
    emissions = [map(float, [d.mean, d.variance]) for d in hmm.emissionDistributions]
    # print init
    # print trans
    # print emissions
    return ghmm.HMMFromMatrices(emissionDomain=domain,
                                distribution=ghmm.GaussianDistribution(domain),
                                A=trans,
                                B=emissions,
                                pi=init)


def ghmm_from_multivariate_continuous_hmm(hmm):
    hmm = deepcopy(hmm)
    domain = ghmm.Float()
    trans = hmm.transitionMatrix.tolist()
    init = hmm.initialProbabilities.tolist()
    emissions = [[d.mean.tolist(), d.variance.flatten().tolist()] for d in hmm.emissionDistributions]
    # print init
    # print trans
    # print emissions
    return ghmm.HMMFromMatrices(emissionDomain=domain,
                                distribution=ghmm.MultivariateGaussianDistribution(domain),
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
