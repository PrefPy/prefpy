# Filename: plackettlucedata.py
# Author: Peter Piech
# Date: 10/2/2015
# Description: Generate Plackett-Luce Model rank data via the command line

import sys
import numpy as np
import scipy.stats

def _generate_pl_dataset(n, m, outfile, useDirichlet):
    """
    Description:
        Generate a Plackett-Luce dataset and
        save it to disk.
    Parameters:
        n:            number of votes to generate
        m:            number of alternatives
        outfile:      open file object to which the dataset is written
        useDirichlet: boolean flag to use the Dirichlet distribution
    """

    gamma, votes = generate_pl_dataset(n, m, useDirichlet)
    outfile.write(str(len(gamma)) + ',' + str(len(votes)) + '\n')
    outfile.write(','.join(map(str, gamma)) + '\n')
    for vote in votes:
        outfile.write(','.join(map(str, vote)) + '\n')
    return (gamma, votes)

def generate_pl_dataset(n, m, useDirichlet=True):
    """
    Description:
        Generate a Plackett-Luce dataset and return the parameters and votes
    Parameters:
        n:            number of votes to generate
        m:            number of alternatives
        useDirichlet: boolean flag to use the Dirichlet distribution
    """
    gamma = None
    if useDirichlet:
        gamma = np.random.dirichlet(np.ones(m))
    else:
        gamma = np.random.rand(m)
        gamma /= np.sum(gamma) # normalize sum to 1.0 (not needed for Dirichlet)
    votes = []
    for i in range(n): # generate vote for every agent
        votes.append(draw_pl_vote(m, gamma))
    return (gamma, votes)

def read_pl_dataset(infile):
    """
    Description:
        Read from disk a Plackett-Luce dataset.
    Parameters:
        infile: open file object from which to read the dataset
    """

    m, n = [int(i) for i in infile.readline().split(',')]
    gamma = np.array([float(f) for f in infile.readline().split(',')])
    if len(gamma) != m:
        infile.close()
        raise ValueError("malformed file: len(gamma) != m")
    votes = []
    i = 0
    for line in infile:
        vote = [int(v) for v in line.split(',')]
        if len(vote) != m:
            infile.close()
            raise ValueError("malformed file: len(vote) != m")
        votes.append(vote)
        i += 1
    infile.close()
    if i != n:
        raise ValueError("malformed file: number of votes != n")
    return (gamma, np.array(votes))

def draw_pl_vote(m, gamma):
    """
    Description:
        Generate a Plackett-Luce vote given the model parameters.
    Parameters:
        m:     number of alternatives
        gamma: parameters of the Plackett-Luce model
    """

    localgamma = np.copy(gamma) # work on a copy of gamma
    localalts = np.arange(m) # enumeration of the candidates
    vote = []
    for j in range(m): # generate position in vote for every alternative

        # transform local gamma into intervals up to 1.0
        localgammaintervals = np.copy(localgamma)
        prev = 0.0
        for k in range(len(localgammaintervals)):
            localgammaintervals[k] += prev
            prev = localgammaintervals[k]

        selection = np.random.random() # pick random number

        # selection will fall into a gamma interval
        for l in range(len(localgammaintervals)): # determine position
            if selection <= localgammaintervals[l]:
                vote.append(localalts[l])
                localgamma = np.delete(localgamma, l) # remove that gamma
                localalts = np.delete(localalts, l) # remove the alternative
                localgamma /= np.sum(localgamma) # renormalize
                break
    return vote

def _generate_mix2pl_dataset(n, m, outfile, useDirichlet=True):
    """
    Description:
        Generate a Mixture of 2 Plackett-Luce models
        dataset and save it to disk.
    Parameters:
        n:            number of votes to generate
        m:            number of alternatives
        outfile:      open file object to which the dataset is written
        useDirichlet: boolean flag to use the Dirichlet distribution
    """
    params, votes = generate_mix2pl_dataset(n, m, useDirichlet)
    outfile.write(str(m) + ',' + str(n) + '\n')
    outfile.write(','.join(map(str, params)) + '\n')
    for vote in votes:
        outfile.write(','.join(map(str, vote)) + '\n')
    return (params, votes)

def read_mix2pl_dataset(infile, numVotes=None):
    """
    Description:
        Read from disk a Mixture of 2 Plackett-Luce models dataset.
    Parameters:
        infile:   open file object from which to read the dataset
        numVotes: number of votes to read from the file or all if None
    """
    m, n = [int(i) for i in infile.readline().split(',')]
    if numVotes is not None and n < numVotes:
        raise ValueError("invalid number of votes to read: exceeds file amount")
    params = np.array([float(f) for f in infile.readline().split(',')])
    if len(params) != (2*m + 1):
        infile.close()
        raise ValueError("malformed file: len(params) != 2*m + 1")
    votes = []
    i = 0
    for line in infile:
        if i > (numVotes - 1):
            break
        vote = [int(v) for v in line.split(',')]
        if len(vote) != m:
            infile.close()
            raise ValueError("malformed file: len(vote) != m")
        votes.append(vote)
        i += 1
    infile.close()
    return (params, np.array(votes))

def generate_mix2pl_dataset(n, m, useDirichlet=True):
    """
    Description:
        Generate a mixture of 2 Plackett-Luce models dataset
        and return the parameters and votes.
    Parameters:
        n:            number of votes to generate
        m:            number of alternatives
        useDirichlet: boolean flag to use the Dirichlet distribution
    """

    alpha = np.random.rand()
    gamma1 = None
    gamma2 = None

    if useDirichlet:
        gamma1 = np.random.dirichlet(np.ones(m))
        gamma2 = np.random.dirichlet(np.ones(m))
    else:
        gamma1 = np.random.rand(m)
        gamma1 /= np.sum(gamma) # normalize sum to 1.0 (not needed for Dirichlet)
        gamma2 = np.random.rand(m)
        gamma2 /= np.sum(gamma)

    votes = []

    for i in range(n):
        vote = None
        draw = np.random.rand()
        if draw <= alpha:
            vote = draw_pl_vote(m, gamma1)
        else: # draw > alpha
            vote = draw_pl_vote(m, gamma2)
        votes.append(vote)
    params = np.hstack((alpha, gamma1, gamma2))
    return (params, votes)

def main(argv):
    if argv is None:
        sys.exit()
    elif len(argv) < 4:
        print("USAGE: python3 " + argv[0] +
              " <n (# agents)> <m (# alternatives)> <output filename> [-U]\n" +
              "  Optional Parameters:\n    -U : use a uniform distribution (default is Dirichlet)")
        sys.exit()

    n = int(argv[1])
    m = int(argv[2])
    outfile = open(argv[3], 'w')
    useDirich = True
    if len(argv) > 4 and argv[4] == "-U":
        useDirich = False

    _generate_pl_dataset(n, m, outfile, useDirich)

    outfile.close()


if __name__ == "__main__":
    main(sys.argv)
