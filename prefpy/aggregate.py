# Base class for all rank aggregation methods

class RankAggregator:
    """
    Base class and methods intended only for use
    in implementing more specific rank aggregation models
    """

    def __init__(self, alts_list):
        """
        Description:
            Initializes the aggregator with the set of alternatives
            and the number of candidates
        Parameters:
            alts_list: the set of integer alternatives (a.k.a candidates)
        """
        self.alts = alts_list
        self.alts_set = set(alts_list)
        self.m = len(alts_list)
        if len(self.alts) != len(self.alts_set):
            raise ValueError("Alternatives must not contain duplicates")
        self.alts_to_ranks = None # Maps alternatives to ranking (projective)
        self.ranks_to_alts = None # Maps rankings to alternatives (surjective)

    def aggregate(self, rankings):
        """
        Description:
            Takes in a set of rankings and returns the aggregate
            parameters according to the rank aggregation model.
        Parameters:
            rankings: a list of tuples where the lowest index of a
                      tuple is the highest rank position
        """
        raise NotImplementedError("Class must be extended to use")

    def get_ranking(self, alt):
        """
        Description:
            Returns the ranking of a given alternative in the
            computed aggregate ranking.  An error is thrown if
            the alternative does not exist.  The ranking is the
            index in the aggregate ranking, which is 0-indexed.
        Parameters:
            alt: the key that represents an alternative
        """
        if self.alts_to_ranks is None:
            raise ValueError("Aggregate ranking must be created first")
        try:
            rank = self.alts_to_ranks[alt]
            return rank
        except KeyError:
            raise KeyError("No alternative \"{}\" found in ".format(str(alt)) +
                           "the aggregate ranking")

    def get_alternatives(self, rank):
        """
        Description:
            Returns the alternative(s) with the given ranking in the
            computed aggregate ranking.  An error is thrown if the
            ranking does not exist.  
        """
        if self.ranks_to_alts is None:
            raise ValueError("Aggregate ranking must be created first")
        try:
            alts = self.ranks_to_alts[rank]
            return alts
        except KeyError:
            raise KeyError("No ranking \"{}\" found in ".format(str(rank)) +
                           "the aggregate ranking")

    def create_rank_dicts(self, alt_scores):
        """
        Description:
            Takes in the scores of the alternatives in the form alt:score and
            generates the dictionaries mapping alternatives to rankings and
            rankings to alternatives.
        Parameters:
            alt_scores: dictionary of the scores of every alternative
        """
        self.alts_to_ranks = dict()
        cur_score = max(alt_scores.values())
        cur_rank = 0
        self.ranks_to_alts = {cur_rank:[]}
        for i in sorted(alt_scores.keys(), key=lambda x: -alt_scores[x]):
            if alt_scores[i] == cur_score:
                self.ranks_to_alts[cur_rank].append(i)
            elif alt_scores[i] < cur_score:
                cur_rank += 1
                cur_score = alt_scores[i]
                self.ranks_to_alts[cur_rank] = [i]
            self.alts_to_ranks[i] = cur_rank
