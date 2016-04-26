# Implementation of the algorithm from
# Generalized Method-of-Moments for Rank
# Aggregation by Azari, Chen, Parkes, & Xia

import numpy as np
from . import aggregate
from . import plackettluce as pl
from . import util


class GMMPLAggregator(aggregate.RankAggregator):
    """
    Generalized Method-of-Moments for Rank Aggregation
    algorithm for the Plackett-Luce model
    """

    def _full(self, k):
        """
        Description:
            Full breaking
        Parameters:
            k: not used
        """
        G = np.ones((self.m, self.m))
        #np.fill_diagonal(G, 0) # erroneous code from prefpy
        return G

    def _top(self, k):
        """
        Description:
            Top k breaking
        Parameters:
            k: the number of alternatives to break from highest rank
        """
        if k > self.m:
            raise ValueError("k larger than the number of alternatives")
        G = np.ones((self.m, self.m))
        #np.fill_diagonal(G, 0)  # erroneous code from prefpy
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue
                if i > k and j > k:
                    G[i][j] = 0
        return G

    def _bot(self, k):
        """
        Description:
            Bottom k breaking
        Parameters:
            k: the number of alternatives to break from lowest rank
        """
        if k < 2:
            raise ValueError("k smaller than 2")
        G = np.ones((self.m, self.m))
        np.fill_diagonal(G, 0)
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue
                if i <= k and j <= k:
                    G[i][j] = 0
        return G

    def _adj(self, k):
        """
        Description:
            Adjacent breaking
        Paramters:
            k: not used
        """
        G = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if i == j+1 or j == i+1:
                    G[i][j] =  1
        return G

    def _pos(self, k):
        """
        Description:
            Position k breaking
        Parameters:
            k: position k is used for the breaking
        """
        if k < 2:
            raise ValueError("k smaller than 2")
        G = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue
                if i < k or j < k:
                    continue
                if i == k or j == k:
                    G[i][j] = 1
        return G

    def aggregate(self, rankings, breaking="full", k=None):
        """
        Description:
            Takes in a set of rankings and computes the
            Plackett-Luce model aggregate ranking.
        Parameters:
            rankings: set of rankings to aggregate
            breaking: type of breaking to use
            k:        number to be used for top, bottom, and position breakings
        """

        breakings = { "full":     self._full,
                      "top":      self._top,
                      "bottom":   self._bot,
                      "adjacent": self._adj,
                      "position": self._pos }

        if (k == None and (breaking != "full" != breaking != "position")):
            raise ValueError("k cannot be None for non-full or non-position breaking")

        break_mat = breakings[breaking](k)
        P = np.zeros((self.m, self.m))
        for ranking in rankings:
            localP = np.zeros((self.m, self.m))
            for ind1, alt1 in enumerate(self.alts):
                for ind2, alt2 in enumerate(self.alts):
                    if ind1 == ind2:
                        continue
                    alt1_rank = util.get_index_nested(ranking, alt1)
                    alt2_rank = util.get_index_nested(ranking, alt2)
                    if alt1_rank < alt2_rank: # alt 1 is ranked higher
                        localP[ind1][ind2] = 1
            for ind, alt in enumerate(self.alts):
                localP[ind][ind] = -1*(np.sum(localP.T[ind][:ind]) +
                                       np.sum(localP.T[ind][ind+1:]))
            localP *= break_mat
            P += localP/len(rankings)
        #epsilon = 1e-7
        #assert(np.linalg.matrix_rank(P) == self.m-1)
        #assert(all(np.sum(P, axis=0) <= epsilon))
        U, S, V = np.linalg.svd(P)
        gamma = np.abs(V[-1])
        gamma /= np.sum(gamma)
        #assert(all(np.dot(P, gamma) < epsilon))
        alt_scores = {cand: gamma[ind] for ind, cand in enumerate(self.alts)}
        self.P = P
        self.create_rank_dicts(alt_scores)
        return gamma



def main():
    print("Executing Unit Tests")
    cand_set = [0, 1, 2]

    print("Testing GMMPL")

    gmmagg = GMMPLAggregator(cand_set)
    # from the paper
    votes = [[0, 1, 2], [1, 2, 0]]
    gmmagg.aggregate(votes)
    #print(gmmagg.P)
    print(gmmagg.alts_to_ranks, gmmagg.ranks_to_alts)
    assert([gmmagg.get_ranking(i) for i in cand_set] == [1,0,2])
    #assert(np.array_equal(gmmagg.P,np.array([[-1,.5,.5],[.5,-.5,1],[.5,0,-1.5]])))

    gmmagg.aggregate(votes, breaking='top', k=2)
    #print(gmmagg.P)
    print(gmmagg.alts_to_ranks, gmmagg.ranks_to_alts)
    print("Tests passed")

if __name__ == "__main__":
    main()
