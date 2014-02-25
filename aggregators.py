# Set of ranking aggregators that, given a set of full rankings output an aggregate ranking
import numpy as np
import scipy as sp
import random


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

class BordaAgg(RankAggBase):
  def aggregate(self, rankings):
    """ Given a set of rankings, computes the aggregate Borda score for each candidate and uses that to create a final aggregate preference ordering """
    self.agg_ctr = dict()
    cand_scores = {i:0 for i in self.cands}
    # For each ranking, go through it and add len - index to candidate scores
    for ranking in rankings:
      for index, equiv_class in enumerate(ranking):
        borda_score = len(ranking) - index
        for cand in equiv_class:
          cand_scores[cand] += borda_score
    print "Candidate scores: ", cand_scores
    # Shitty hack to make equivalence classes (i.e. breaking ties by not breaking ties)
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

class GMMPLAgg(RankAggBase):	
  def aggregate(self, rankings):
    """ Given a set of rankings, computes the Placket-Luce model for preferences """
    pass


if __name__ == "__main__":
  cand_set = set(['a','b','c'])
  votes = [[tuple('a'),tuple('b')]]
  bagg = BordaAgg(cand_set)
  bagg.aggregate(votes)
  print bagg.agg_ctr, bagg.agg_rtc



           

