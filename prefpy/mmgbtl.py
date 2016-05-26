# Implementation of algorithm (3) from
# MM Algorithms for Generalized Bradley-Terry Models
# by David R. Hunter, 2004

import numpy as np
from . import aggregate
from . import plackettluce as pl
from . import util


class MMPLAggregator(aggregate.RankAggregator):
    """
    Minorization-Maximization Rank Aggregation
    algorithm for the Plackett-Luce model
    """

    def aggregate(self, rankings, epsilon, max_iters):
        """
        Description:
            Minorization-Maximization algorithm which returns an
            estimate of the ground-truth parameters, gamma for
            the given data.
        Parameters:
            rankings:  set of rankings to aggregate
            epsilon:   convergence condition value, set to None for iteration only
            max_iters: maximum number of iterations of MM algorithm
        """

        # compute the matrix w, the numbers of pairwise wins:
        w = np.zeros((self.m, self.m))
        for ranking in rankings:
            localw = np.zeros((self.m, self.m))
            for ind1, alt1 in enumerate(self.alts):
                for ind2, alt2 in enumerate(self.alts):
                    if ind1 == ind2:
                        continue
                    alt1_rank = util.get_index_nested(ranking, alt1)
                    alt2_rank = util.get_index_nested(ranking, alt2)
                    if alt1_rank < alt2_rank: # alt 1 is ranked higher
                        localw[ind1][ind2] = 1
            w += localw
        W = w.sum(axis=1)

        # gamma_t is the value of gamma at time = t
        # gamma_t1 is the value of gamma at time t = t+1 (the next iteration)
        # initial arbitrary value for gamma:
        gamma_t = np.ones(self.m) / self.m
        gamma_t1 = np.empty(self.m)

        for f in range(max_iters):

            for i in range(self.m):
                s = 0 # sum of updating function
                for j in range(self.m):
                    if j != i:
                        s += (w[j][i] + w[i][j]) / (gamma_t[i]+gamma_t[j])

                gamma_t1[i] = W[i] / s

            gamma_t1 /= np.sum(gamma_t1)

            if epsilon != None and np.all(np.absolute(gamma_t1 - gamma_t) < epsilon):
                alt_scores = {cand: gamma_t1[ind] for ind, cand in enumerate(self.alts)}
                self.create_rank_dicts(alt_scores)
                return gamma_t1 # convergence reached before max_iters

            gamma_t = gamma_t1 # update gamma_t for the next iteration
        alt_scores = {cand: gamma_t1[ind] for ind, cand in enumerate(self.alts)}
        self.create_rank_dicts(alt_scores)
        return gamma_t1


def main():
    """Driver function for the computation of the MM algorithm"""

    # test example below taken from GMMRA by Azari, Chen, Parkes, & Xia
    cand_set = [0, 1, 2]
    votes = [[0, 1, 2], [1, 2, 0]]

    mmagg = MMPLAggregator(cand_set)
    gamma = mmagg.aggregate(votes, epsilon=1e-7, max_iters=20)
    print(mmagg.alts_to_ranks, mmagg.ranks_to_alts)
    assert([mmagg.get_ranking(i) for i in cand_set] == [1,0,2])
    print(gamma)

if __name__ == "__main__":
    main()
