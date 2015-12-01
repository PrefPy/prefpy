# Filename: plackettlucedata.py
# Author: Peter Piech
# Date: 10/2/2015
# Description: Generate Plackett-Luce Model rank data via the command line

import sys
import numpy as np
import scipy.stats

def _generate_pl_votes(n, m, outfile, useDirichlet):
    gamma,  votes = generate_pl_votes(n, m, useDirichlet)
    outfile.write(str(len(votes)) + ',' + str(len(gamma)) + '\n')
    outfile.write(','.join(map(str, gamma)) + '\n')
    for vote in votes:
        outfile.write(','.join(map(str, vote)) + '\n')

def generate_pl_votes(n, m, useDirichlet=True):
    alts = np.array(range(m)) # TODO: change to `alts = np.arange(m)`

    gamma = np.empty(m)

    if useDirichlet:
        gamma = np.random.dirichlet(np.ones(m))
    else:
        gamma = np.random.rand(m)
        gamma /= np.sum(gamma) # normalize sum to 1.0 (not needed for Dirichlet)

    votes = []

    for i in range(n): # generate vote for every agent

        localgamma = np.copy(gamma) # reset local gamma to global value
        localalts = np.copy(alts) # reset alternatives to global list
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

        votes.append(vote)

#    # Methods below do not provide desired results:
#    for i in range(n):
#        vote = np.array([np.random.gamma(g, 1) for g in gamma]) # => very high MSE, close to 0 Kendall correlation
#        vote = np.array([np.random.gumbel(np.log(g), 1) for g in gamma]) # => extremely high MSE, negative Kendall correlation
#        vote /= np.sum(vote)
#        vote = (scipy.stats.rankdata(-1*vote, method='ordinal')-1).astype(int)
#        votes.append(vote)

    return (gamma, votes)

def read_pl_votes(infile):
    n, m = [int(i) for i in infile.readline().split(',')]
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

    _generate_pl_votes(n, m, outfile, useDirich)

    outfile.close()


if __name__ == "__main__":
    main(sys.argv)
