# Set of ranking aggregators that, given a set of full rankings output an aggregate ranking
import numpy as np
import scipy as sp
import random

def get_index_nested(x, i): 
  for ind in range(len(x)): 
    if i in x[ind]: 
      return ind
  return -1

class RankAggBase(object):
  def __init__(self, cand_set):
    """ Initializes the aggregator by a set of all possible candidates and stores this along with the number of candidates"""
    self.m = len(cand_set)
    self.cands = cand_set
    self.agg_ctr = None # Mapping of candidates to ranking (projective)
    self.agg_rtc = None # Mapping of rankings to candidates (surjective)
  
  def aggregate(self, rankings):
    """ Given a set of rankings, produce the aggregate ranking. Each ranking is a list of tuples (to accomodate equivalence classes) with the lowest index having the highest rank """
    # When implementing, set the aggregate in self.agg_ctr and self.agg_rtc correctly
    raise NotImplementedError("Do not use this class")

  def get_ranking(self, cand):
    """ Gets the ranking of a given candidate in the aggregate ranking. If the candidate is not in the aggregate, throws an error. Also throws an error if there is no aggregate. This ranking is the index in the aggregate, which is 0-indexed """
    if self.agg_ctr == None:
      raise ValueError("No aggregate ranking yet. Run the aggregate method on some rankings first")
    try:
      rank = self.agg_ctr[cand]
      return rank
    except KeyError as ke:
      print "No candidate by %s found in the aggregate ranking" % str(cand)
      raise ke

  def get_candidates(self, rank):
    """ Gets the candidate(s) corresponding to the given ranking. Raises an error is agg_rtc is not present or if the ranking is not in it. """
    if self.agg_rtc == None:
      raise ValueError("No aggregate ranking yet. Run the aggregate method on some rankings first")
    try:
      candidates = self.agg_rtc[rank]
      return candidates
    except KeyError as ke:
      print "The ranking %d is not in the aggregate ranking" % rank
      raise ke
  
  def create_rank_dicts(self, cand_scores):
    """ Takes the candidate scores in form cand:score and returns a dictionary of rankings to candidates and candidates to rankings """
    # Shitty hack to make equivalence classes (i.e. breaking ties by not breaking ties)
    self.agg_ctr = dict()
    cur_score = max(cand_scores.values())
    cur_rank = 1
    self.agg_rtc = {cur_rank:[]}
    for i in sorted(cand_scores.keys(), key=lambda x: -cand_scores[x]):
      if cand_scores[i] == cur_score:
        self.agg_rtc[cur_rank].append(i)
      elif cand_scores[i] < cur_score:
        cur_rank += 1
        cur_score = cand_scores[i]
        self.agg_rtc[cur_rank] = [i]
      self.agg_ctr[i] = cur_rank



class BordaAgg(RankAggBase):
  def aggregate(self, rankings):
    """ Given a set of rankings, computes the aggregate Borda score for each candidate and uses that to create a final aggregate preference ordering """
    cand_scores = {i:0 for i in self.cands}
    # For each ranking, go through it and add len - index to candidate scores
    for ranking in rankings:
      for index, equiv_class in enumerate(ranking):
        borda_score = len(ranking) - index
        for cand in equiv_class:
          cand_scores[cand] += borda_score
    self.create_rank_dicts(cand_scores)
          
class GMMPLAgg(RankAggBase):  
  def aggregate(self, rankings, breaking='full', K=None):
    """ Given a set of rankings, computes the Placket-Luce model for preferences """
    # Ignore breakings for now
    n_cands = len(self.cands)
    breaking = np.ones((n_cands,n_cands))
    cands_list = list(self.cands)
    # So this is kinda hacky, but essentially we want a mapping of index -> candidate for the matrix, which can be arbitrary as long as it's consistent
    P = np.zeros((n_cands,n_cands))
    for ranking in rankings:
      localP = np.zeros((n_cands,n_cands))
      for ind1, cand1 in enumerate(cands_list):
        for ind2, cand2 in enumerate(cands_list):
          if ind1 == ind2:
            continue
          cand1_rank = get_index_nested(ranking, cand1)
          cand2_rank = get_index_nested(ranking, cand2)
          if cand1_rank < cand2_rank: # i.e. cand 1 is ranked higher
            localP[ind1][ind2] = 1
      localP *= breaking
      for ind, cand in enumerate(cands_list):
        localP[ind][ind] = -1*(np.sum(localP.T[ind][:ind]) + np.sum(localP.T[ind][ind+1:])) # quick and dirty way to do sum w/o i
      P += localP
    eps = 1e-7 # Not really 0, but close enough?
    assert(np.linalg.matrix_rank(P) == self.m-1)
    assert(all(np.sum(P, axis=0) <= eps))
    U, S, V = np.linalg.svd(P)
    gamma = np.abs(V[-1])
    assert(all(np.dot(P, gamma) < eps))  
    cand_scores = {cand:gamma[ind] for ind, cand in enumerate(cands_list)}
    self.create_rank_dicts(cand_scores)


if __name__ == "__main__":
  cand_set = set(['a','b','c'])
  votes = [[tuple('a'),tuple('b')]]
  bagg = GMMPLAgg(cand_set)
  bagg.aggregate(votes)
  print bagg.agg_ctr, bagg.agg_rtc



           

