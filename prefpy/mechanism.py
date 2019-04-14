"""
Authors: Kevin J. Hwang
         Jun Wang
         Tyler Shepherd
"""
import io
import math
import time
from numpy import *
import itertools
from preference import Preference
from profile import Profile
import copy
import sys
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
from queue import PriorityQueue


class Mechanism():
    """
    The parent class for all mechanisms. This class should not be constructed directly. All child
    classes are expected to contain the following variable(s).

    :ivar bool maximizeCandScore: True if the mechanism requires winners to maximize their score
        and if False otherwise.
    """

    def getWinners(self, profile):
        """
        Returns a list of all winning candidates given an election profile. This function assumes
        that getCandScoresMap(profile) is implemented for the child Mechanism class.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        candScores = self.getCandScoresMap(profile)

        # Check whether the winning candidate is the candidate that maximizes the score or
        # minimizes it.
        if self.maximizeCandScore == True:
            bestScore = max(candScores.values())
        else:
            bestScore = min(candScores.values())

        # Create a list of all candidates with the winning score and return it.
        winners = []
        for cand in candScores.keys():
            if candScores[cand] == bestScore:
                winners.append(cand)
        return winners

    def getRanking(self, profile):
        """
        Returns a list of lists that orders all candidates in tiers from best to worst given an
        election profile. This function assumes that getCandScoresMap(profile) is implemented for
        the child Mechanism class.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # We generate a map that associates each score with the candidates that have that acore.
        candScoresMap = self.getCandScoresMap(profile)
        reverseCandScoresMap = dict()
        for key, value in candScoresMap.items():
            if value not in reverseCandScoresMap.keys():
                reverseCandScoresMap[value] = [key]
            else:
                reverseCandScoresMap[value].append(key)

        # We sort the scores by either decreasing order or increasing order.
        if self.maximizeCandScore == True:
            sortedCandScores = sorted(reverseCandScoresMap.keys(), reverse=True)
        else:
            sortedCandScores = sorted(reverseCandScoresMap.keys())

        # We put the candidates into our ranking based on the order in which their score appears
        ranking = []
        for candScore in sortedCandScores:
            currRanking = []
            for cand in reverseCandScoresMap[candScore]:
                currRanking.append(cand)
            ranking.append(currRanking)

        # Right now we return a list that contains the ranking list. This is for future extensions.
        results = []
        results.append(ranking)

        return results


class MechanismPosScoring(Mechanism):
    """
    The positional scoring mechanism. This class is the parent class for several mechanisms. This
    can also be constructed directly. All child classes are expected to implement the
    getScoringVector() method.

    :ivar list<int> scoringVector: A list of integers (or floats) that give the scores assigned to
        each position in a ranking from first to last.
    """

    def __init__(self, scoringVector):
        self.maximizeCandScore = True
        self.scoringVector = scoringVector

    def isProfileValid(self, profile):
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc":
            return False
        return True

    def getScoringVector(self, profile):
        """
        Returns the scoring vector. This function is called by getCandScoresMap().

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Check to make sure that the scoring vector contains a score for every possible rank in a
        # ranking.
        if len(self.scoringVector) != profile.numCands:
            print("ERROR: scoring vector is not the correct length")
            exit()

        return self.scoringVector

    def getCandScoresMap(self, profile):
        """
        Returns a dictonary that associates the integer representation of each candidate with the
        score they recieved in the profile.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc":
            print("ERROR: unsupported election type")
            exit()

        # Initialize our dictionary so that all candidates have a score of zero.
        candScoresMap = dict()
        for cand in profile.candMap.keys():
            candScoresMap[cand] = 0.0

        rankMaps = profile.getRankMaps()
        rankMapCounts = profile.getPreferenceCounts()
        scoringVector = self.getScoringVector(profile)

        # Go through the rankMaps of the profile and increment each candidates score appropriately.
        for i in range(0, len(rankMaps)):
            rankMap = rankMaps[i]
            rankMapCount = rankMapCounts[i]
            for cand in rankMap.keys():
                candScoresMap[cand] += scoringVector[rankMap[cand] - 1] * rankMapCount

        # print("candScoresMap=", candScoresMap)
        return candScoresMap

    def getMov(self, profile):
        """
        Returns an integer that is equal to the margin of victory of the election profile.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        # from . import mov
        import mov
        return mov.MoVScoring(profile, self.getScoringVector(profile))


class MechanismPlurality(MechanismPosScoring):
    """
    The plurality mechanism. This inherits from the positional scoring mechanism.
    """

    def __init__(self):
        self.maximizeCandScore = True

    def getScoringVector(self, profile):
        """
        Returns the scoring vector [1,0,0,...,0]. This function is called by getCandScoresMap()
        which is implemented in the parent class.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        scoringVector = []
        scoringVector.append(1)
        for i in range(1, profile.numCands):
            scoringVector.append(0)
        return scoringVector


class MechanismVeto(MechanismPosScoring):
    """
    The veto mechanism. This inherits from the positional scoring mechanism.
    """

    def __init__(self):
        self.maximizeCandScore = True

    def getScoringVector(self, profile):
        """
        Returns the scoring vector [1,1,1,...,0]. This function is called by getCandScoresMap()
        which is implemented in the parent class.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        numTiers = len(set(profile.getRankMaps()[0].values()))
        scoringVector = []
        for i in range(0, numTiers - 1):
            scoringVector.append(1)
        for i in range(numTiers - 1, profile.numCands):
            scoringVector.append(0)
        return scoringVector


class MechanismBorda(MechanismPosScoring):
    """
    The Borda mechanism. This inherits from the positional scoring mechanism.
    """

    def __init__(self):
        self.maximizeCandScore = True

    def getScoringVector(self, profile):
        """
        Returns the scoring vector [m-1,m-2,m-3,...,0] where m is the number of candidates in the
        election profile. This function is called by getCandScoresMap() which is implemented in the
        parent class.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        scoringVector = []
        score = profile.numCands - 1
        for i in range(0, profile.numCands):
            scoringVector.append(score)
            score -= 1
        return scoringVector


class MechanismKApproval(MechanismPosScoring):
    """
    The top-k mechanism. This inherits from the positional scoring mechanism.

    :ivar int k: The number of positions that recieve a score of 1.
    """

    def __init__(self, k):
        self.maximizeCandScore = True
        self.k = k

    def getScoringVector(self, profile):
        """
        Returns a scoring vector such that the first k candidates recieve 1 point and all others
        recive 0  This function is called by getCandScoresMap() which is implemented in the parent
        class.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        if self.k > profile.numCands:
            self.k = profile.numCands

        scoringVector = []
        for i in range(0, self.k):
            scoringVector.append(1)
        for i in range(self.k, profile.numCands):
            scoringVector.append(0)
        return scoringVector


class MechanismSimplifiedBucklin(Mechanism):
    """
    The simplified Bucklin mechanism.
    """

    def __init__(self):
        self.maximizeCandScore = False

    def getCandScoresMap(self, profile):
        """
        Returns a dictionary that associates integer representations of each candidate with their
        Bucklin score.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc":
            print("ERROR: unsupported profile type")
            exit()

        bucklinScores = dict()
        rankMaps = profile.getRankMaps()
        preferenceCounts = profile.getPreferenceCounts()
        for cand in profile.candMap.keys():

            # We keep track of the number of times a candidate is ranked in the first t positions.
            numTimesRanked = 0

            # We increase t in increments of 1 until we find t such that the candidate is ranked in the
            # first t positions in at least half the votes.
            for t in range(1, profile.numCands + 1):
                for i in range(0, len(rankMaps)):
                    if (rankMaps[i][cand] == t):
                        numTimesRanked += preferenceCounts[i]
                if numTimesRanked >= math.ceil(float(profile.numVoters) / 2):
                    bucklinScores[cand] = t
                    break

        return bucklinScores

    def getMov(self, profile):
        """
        Returns an integer that is equal to the margin of victory of the election profile.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        # from . import mov
        import mov
        return mov.MoVSimplifiedBucklin(profile)


class MechanismCopeland(Mechanism):
    """
    The Copeland mechanism.
    """

    def __init__(self, alpha):
        self.maximizeCandScore = True
        self.alpha = 0.5

    def getCandScoresMap(self, profile):
        """
        Returns a dictionary that associates integer representations of each candidate with their
        Copeland score.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are
        # allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc":
            print("ERROR: unsupported election type")
            exit()

        # Initialize each Copeland score as 0.0.
        copelandScores = dict()
        for cand in profile.candMap.keys():
            copelandScores[cand] = 0.0

        preferenceCounts = profile.getPreferenceCounts()

        # For each pair of candidates, calculate the number of votes in which one beat the other.
        wmgMap = profile.getWmg()
        for cand1, cand2 in itertools.combinations(wmgMap.keys(), 2):
            if cand2 in wmgMap[cand1].keys():
                if wmgMap[cand1][cand2] > 0:
                    copelandScores[cand1] += 1.0
                elif wmgMap[cand1][cand2] < 0:
                    copelandScores[cand2] += 1.0

                # If a pair of candidates is tied, we add alpha to their score for each vote.
                else:
                    copelandScores[cand1] += self.alpha
                    copelandScores[cand2] += self.alpha

        return copelandScores


class MechanismMaximin(Mechanism):
    """
    The maximin mechanism.
    """

    def __init__(self):
        self.maximizeCandScore = True

    def getCandScoresMap(self, profile):
        """
        Returns a dictionary that associates integer representations of each candidate with their
        maximin score.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are
        # allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc":
            print("ERROR: unsupported election type")
            exit()

        wmg = profile.getWmg()

        # Initialize the maximin score for each candidate as infinity.
        maximinScores = dict()
        for cand in wmg.keys():
            maximinScores[cand] = float("inf")

        # For each pair of candidates, calculate the number of times each beats the other.
        for cand1, cand2 in itertools.combinations(wmg.keys(), 2):
            if cand2 in wmg[cand1].keys():
                maximinScores[cand1] = min(maximinScores[cand1], wmg[cand1][cand2])
                maximinScores[cand2] = min(maximinScores[cand2], wmg[cand2][cand1])

        return maximinScores


class MechanismSchulze(Mechanism):
    """
    The Schulze mechanism.
    """

    def __init__(self):
        self.maximizeCandScore = True

    def computeStrongestPaths(self, profile, pairwisePreferences):
        """
        Returns a two-dimensional dictionary that associates every pair of candidates, cand1 and
        cand2, with the strongest path from cand1 to cand2.

        :ivar Profile profile: A Profile object that represents an election profile.
        :ivar dict<int,dict<int,int>> pairwisePreferences: A two-dimensional dictionary that
            associates every pair of candidates, cand1 and cand2, with number of voters who prefer
            cand1 to cand2.
        """
        cands = profile.candMap.keys()
        numCands = len(cands)

        # Initialize the two-dimensional dictionary that will hold our strongest paths.
        strongestPaths = dict()
        for cand in cands:
            strongestPaths[cand] = dict()

        for i in range(1, numCands + 1):
            for j in range(1, numCands + 1):
                if (i == j):
                    continue
                if pairwisePreferences[i][j] > pairwisePreferences[j][i]:
                    strongestPaths[i][j] = pairwisePreferences[i][j]
                else:
                    strongestPaths[i][j] = 0

        for i in range(1, numCands + 1):
            for j in range(1, numCands + 1):
                if (i == j):
                    continue
                for k in range(1, numCands + 1):
                    if (i == k or j == k):
                        continue
                    strongestPaths[j][k] = max(strongestPaths[j][k], min(strongestPaths[j][i], strongestPaths[i][k]))

        return strongestPaths

    def computePairwisePreferences(self, profile):
        """
        Returns a two-dimensional dictionary that associates every pair of candidates, cand1 and
        cand2, with number of voters who prefer cand1 to cand2.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        cands = profile.candMap.keys()

        # Initialize the two-dimensional dictionary that will hold our pairwise preferences.
        pairwisePreferences = dict()
        for cand in cands:
            pairwisePreferences[cand] = dict()
        for cand1 in cands:
            for cand2 in cands:
                if cand1 != cand2:
                    pairwisePreferences[cand1][cand2] = 0

        for preference in profile.preferences:
            wmgMap = preference.wmgMap
            for cand1, cand2 in itertools.combinations(cands, 2):

                # If either candidate was unranked, we assume that they are lower ranked than all
                # ranked candidates.
                if cand1 not in wmgMap.keys():
                    if cand2 in wmgMap.keys():
                        pairwisePreferences[cand2][cand1] += 1 * preference.count
                elif cand2 not in wmgMap.keys():
                    if cand1 in wmgMap.keys():
                        pairwisePreferences[cand1][cand2] += 1 * preference.count

                elif wmgMap[cand1][cand2] == 1:
                    pairwisePreferences[cand1][cand2] += 1 * preference.count
                elif wmgMap[cand1][cand2] == -1:
                    pairwisePreferences[cand2][cand1] += 1 * preference.count

        return pairwisePreferences

    def getCandScoresMap(self, profile):
        """
        Returns a dictionary that associates integer representations of each candidate with the
        number of other candidates for which her strongest path to the other candidate is greater
        than the other candidate's stronget path to her.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        cands = profile.candMap.keys()
        pairwisePreferences = self.computePairwisePreferences(profile)
        strongestPaths = self.computeStrongestPaths(profile, pairwisePreferences)

        # For each candidate, determine how many times p[E,X] >= p[X,E] using a variant of the
        # Floyd-Warshall algorithm.
        betterCount = dict()
        for cand in cands:
            betterCount[cand] = 0
        for cand1 in cands:
            for cand2 in cands:
                if cand1 == cand2:
                    continue
                if strongestPaths[cand1][cand2] >= strongestPaths[cand2][cand1]:
                    betterCount[cand1] += 1

        return betterCount


def getKendallTauScore(myResponse, otherResponse):
    """
    Returns the Kendall Tau Score
    """
    # variables
    kt = 0
    list1 = myResponse.values()
    list2 = otherResponse.values()

    if len(list1) <= 1:
        return kt

    # runs through list1
    for itr1 in range(0, len(list1) - 1):
        # runs through list2
        for itr2 in range(itr1 + 1, len(list2)):
            # checks if there is a discrepancy. If so, adds
            if ((list1[itr1] > list1[itr2]
                 and list2[itr1] < list2[itr2])
                or (list1[itr1] < list1[itr2]
                    and list2[itr1] > list2[itr2])):
                kt += 1
    # normalizes between 0 and 1
    kt = (kt * 2) / (len(list1) * (len(list1) - 1))

    # returns found value
    return kt


class MechanismSTV():
    """
    The STV mechanism.
    """

    def STVwinners(self, profile):
        elecType = profile.getElecType()
        if elecType == "soc" or elecType == "csv":
            return self.STVsocwinners(profile)
        elif elecType == "toc":
            return self.STVtocwinners(profile)
        else:
            print("ERROR: unsupported profile type")
            exit()

    def STVsocwinners(self, profile):
        """
        Returns an integer list that represents all possible winners of a profile under STV rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        ordering = profile.getOrderVectors()
        prefcounts = profile.getPreferenceCounts()
        m = profile.numCands

        if min(ordering[0]) == 0:
            startstate = set(range(m))
        else:
            startstate = set(range(1, m + 1))

        ordering, startstate = self.preprocessing(ordering, prefcounts, m, startstate)
        m_star = len(startstate)
        known_winners = set()
        # ----------Some statistics--------------
        hashtable2 = set()

        # push the node of start state into the priority queue
        root = Node(value=startstate)
        stackNode = []
        stackNode.append(root)

        while stackNode:
            # ------------pop the current node-----------------
            node = stackNode.pop()
            # -------------------------------------------------
            state = node.value.copy()

            # use heuristic to delete all the candidates which satisfy the following condition

            # goal state 1: if the state set contains only 1 candidate, then stop
            if len(state) == 1 and list(state)[0] not in known_winners:
                known_winners.add(list(state)[0])
                continue
            # goal state 2 (pruning): if the state set is subset of the known_winners set, then stop
            if state <= known_winners:
                continue
            # ----------Compute plurality score for the current remaining candidates--------------
            plural_score = self.get_plurality_scores3(prefcounts, ordering, state, m_star)
            minscore = min(plural_score.values())
            for to_be_deleted in state:
                if plural_score[to_be_deleted] == minscore:
                    child_state = state.copy()
                    child_state.remove(to_be_deleted)
                    tpc = tuple(sorted(child_state))
                    if tpc in hashtable2:
                        continue
                    else:
                        hashtable2.add(tpc)
                        child_node = Node(value=child_state)
                        stackNode.append(child_node)

        return sorted(known_winners)

    def STVtocwinners(self, profile):
        """
        Returns an integer list that represents all possible winners of a profile under STV rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        ordering = profile.getOrderVectors()
        prefcounts = profile.getPreferenceCounts()
        len_prefcounts = len(prefcounts)
        m = profile.numCands
        rankmaps = profile.getRankMaps()
        if min(ordering[0]) == 0:
            startstate = set(range(m))
        else:
            startstate = set(range(1, m + 1))

        # ordering, startstate = self.preprocessing(ordering, prefcounts, m, startstate)
        # m_star = len(startstate)
        known_winners = set()
        # ----------Some statistics--------------
        hashtable2 = set()
        root = Node(value=startstate)
        stackNode = []
        stackNode.append(root)

        while stackNode:
            # ------------pop the current node-----------------
            node = stackNode.pop()
            # -------------------------------------------------
            state = node.value.copy()

            # use heuristic to delete all the candidates which satisfy the following condition

            # goal state 1: if the state set contains only 1 candidate, then stop
            if len(state) == 1 and list(state)[0] not in known_winners:
                known_winners.add(list(state)[0])
                continue
            # goal state 2 (pruning): if the state set is subset of the known_winners set, then stop
            if state <= known_winners:
                continue
            # ----------Compute plurality score for the current remaining candidates--------------
            plural_score = self.get_plurality_scores4(prefcounts, rankmaps, state)
            minscore = min(plural_score.values())
            for to_be_deleted in state:
                if plural_score[to_be_deleted] == minscore:
                    child_state = state.copy()
                    child_state.remove(to_be_deleted)
                    tpc = tuple(sorted(child_state))
                    if tpc in hashtable2:
                        continue
                    else:
                        hashtable2.add(tpc)
                        child_node = Node(value=child_state)
                        stackNode.append(child_node)

        return sorted(known_winners)

    def preprocessing(self, ordering, prefcounts, m, startstate):
        plural_score = self.get_plurality_scores3(prefcounts, ordering, startstate, m)
        state = set([key for key, value in plural_score.items() if value != 0])
        ordering = self.construct_ordering(ordering, prefcounts, state)
        plural_score = dict([(key, value) for key, value in plural_score.items() if value != 0])
        minscore = min(plural_score.values())
        to_be_deleted = [key for key, value in plural_score.items() if value == minscore]
        if len(to_be_deleted) > 1:
            return ordering, state
        else:
            while len(to_be_deleted) == 1 and len(state) > 1:
                state.remove(to_be_deleted[0])
                plural_score = self.get_plurality_scores3(prefcounts, ordering, state, m)
                minscore = min(plural_score.values())
                to_be_deleted = [key for key, value in plural_score.items() if value == minscore]
            ordering = self.construct_ordering(ordering, prefcounts, state)
            return ordering, state

    def construct_ordering(self, ordering, prefcounts, state):
        new_ordering = []
        for i in range(len(prefcounts)):
            new_ordering.append([x for x in ordering[i] if x in state])
        return new_ordering

    def get_plurality_scores3(self, prefcounts, ordering, state, m):
        plural_score = {}
        plural_score = plural_score.fromkeys(state, 0)
        for i in range(len(prefcounts)):
            for j in range(m):
                if ordering[i][j] in state:
                    plural_score[ordering[i][j]] += prefcounts[i]
                    break
        return plural_score

    def get_plurality_scores4(self, prefcounts, rankmaps, state):
        plural_score = {}
        plural_score = plural_score.fromkeys(state, 0)
        for i in range(len(prefcounts)):
            temp = list(filter(lambda x: x[0] in state, list(rankmaps[i].items())))
            min_value = min([value for key, value in temp])
            for j in state:
                if rankmaps[i][j] == min_value:
                    plural_score[j] += prefcounts[i]

        return plural_score


class MechanismBaldwin():
    """
    The Baldwin mechanism.
    """

    def baldwin_winners(self, profile):
        elecType = profile.getElecType()
        if elecType == "soc" or elecType == "csv":
            return self.baldwinsoc_winners(profile)
        elif elecType == "toc":
            return self.baldwintoc_winners(profile)
        else:
            print("ERROR: unsupported profile type")
            exit()

    def baldwinsoc_winners(self, profile):
        """
        Returns an integer list that represents all possible winners of a profile under baldwin rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        ordering = profile.getOrderVectors()
        m = profile.numCands
        prefcounts = profile.getPreferenceCounts()
        if min(ordering[0]) == 0:
            startstate = set(range(m))
        else:
            startstate = set(range(1, m + 1))
        wmg = self.getWmg2(prefcounts, ordering, startstate, normalize=False)
        known_winners = set()
        # ----------Some statistics--------------
        hashtable2 = set()

        # push the node of start state into the priority queue
        root = Node(value=startstate)
        stackNode = []
        stackNode.append(root)

        while stackNode:
            # ------------pop the current node-----------------
            node = stackNode.pop()
            # -------------------------------------------------
            state = node.value.copy()

            # goal state 1: if the state set contains only 1 candidate, then stop
            if len(state) == 1 and list(state)[0] not in known_winners:
                known_winners.add(list(state)[0])
                continue
            # goal state 2 (pruning): if the state set is subset of the known_winners set, then stop
            if state <= known_winners:
                continue
            # ----------Compute plurality score for the current remaining candidates--------------
            plural_score = dict()
            for cand in state:
                plural_score[cand] = 0
            for cand1, cand2 in itertools.permutations(state, 2):
                plural_score[cand1] += wmg[cand1][cand2]

            # if current state satisfies one of the 3 goal state, continue to the next loop

            # After using heuristics, generate children and push them into priority queue
            # frontier = [val for val in known_winners if val in state] + list(set(state) - set(known_winners))

            minscore = min(plural_score.values())
            for to_be_deleted in state:
                if plural_score[to_be_deleted] == minscore:
                    child_state = state.copy()
                    child_state.remove(to_be_deleted)
                    tpc = tuple(sorted(child_state))
                    if tpc in hashtable2:
                        continue
                    else:
                        hashtable2.add(tpc)
                        child_node = Node(value=child_state)
                        stackNode.append(child_node)
        return sorted(known_winners)

    def baldwintoc_winners(self, profile):
        """
        Returns an integer list that represents all possible winners of a profile under baldwin rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        ordering = profile.getOrderVectors()
        m = profile.numCands
        prefcounts = profile.getPreferenceCounts()

        rankmaps = profile.getRankMaps()
        if min(ordering[0]) == 0:
            startstate = set(range(m))
        else:
            startstate = set(range(1, m + 1))
        wmg = self.getWmg3(prefcounts, rankmaps, startstate, normalize=False)
        known_winners = set()
        # ----------Some statistics--------------
        hashtable2 = set()

        # push the node of start state into the priority queue
        root = Node(value=startstate)
        stackNode = []
        stackNode.append(root)

        while stackNode:
            # ------------pop the current node-----------------
            node = stackNode.pop()
            # -------------------------------------------------
            state = node.value.copy()

            # goal state 1: if the state set contains only 1 candidate, then stop
            if len(state) == 1 and list(state)[0] not in known_winners:
                known_winners.add(list(state)[0])
                continue
            # goal state 2 (pruning): if the state set is subset of the known_winners set, then stop
            if state <= known_winners:
                continue
            # ----------Compute plurality score for the current remaining candidates--------------
            plural_score = dict()
            for cand in state:
                plural_score[cand] = 0
            for cand1, cand2 in itertools.permutations(state, 2):
                plural_score[cand1] += wmg[cand1][cand2]

            # if current state satisfies one of the 3 goal state, continue to the next loop

            # After using heuristics, generate children and push them into priority queue
            # frontier = [val for val in known_winners if val in state] + list(set(state) - set(known_winners))
            childbranch = 0
            minscore = min(plural_score.values())
            for to_be_deleted in state:
                if plural_score[to_be_deleted] == minscore:
                    child_state = state.copy()
                    child_state.remove(to_be_deleted)
                    tpc = tuple(sorted(child_state))
                    if tpc in hashtable2:
                        continue
                    else:
                        hashtable2.add(tpc)
                        child_node = Node(value=child_state)
                        stackNode.append(child_node)
        return sorted(known_winners)

    def getWmg2(self, prefcounts, ordering, state, normalize=False):
        """
        Generate a weighted majority graph that represents the whole profile. The function will
        return a two-dimensional dictionary that associates integer representations of each pair of
        candidates, cand1 and cand2, with the number of times cand1 is ranked above cand2 minus the
        number of times cand2 is ranked above cand1.

        :ivar bool normalize: If normalize is True, the function will return a normalized graph
            where each edge has been divided by the value of the largest edge.
        """

        # Initialize a new dictionary for our final weighted majority graph.
        wmgMap = dict()
        for cand in state:
            wmgMap[cand] = dict()
        for cand1, cand2 in itertools.combinations(state, 2):
            wmgMap[cand1][cand2] = 0
            wmgMap[cand2][cand1] = 0

        # Go through the wmgMaps and increment the value of each edge in our final graph with the
        # edges in each of the wmgMaps. We take into account the number of times that the vote
        # occured.
        for i in range(0, len(prefcounts)):
            for cand1, cand2 in itertools.combinations(ordering[i], 2):  # --------------------------
                wmgMap[cand1][cand2] += prefcounts[i]

        # By default, we assume that the weighted majority graph should not be normalized. If
        # desired, we normalize by dividing each edge by the value of the largest edge.
        if normalize == True:
            maxEdge = float('-inf')
            for cand in wmgMap.keys():
                maxEdge = max(maxEdge, max(wmgMap[cand].values()))
            for cand1 in wmgMap.keys():
                for cand2 in wmgMap[cand1].keys():
                    wmgMap[cand1][cand2] = float(wmgMap[cand1][cand2]) / maxEdge

        return wmgMap

    def getWmg3(self, prefcounts, rankmaps, state, normalize=False):
        """
        Generate a weighted majority graph that represents the whole profile. The function will
        return a two-dimensional dictionary that associates integer representations of each pair of
        candidates, cand1 and cand2, with the number of times cand1 is ranked above cand2 minus the
        number of times cand2 is ranked above cand1.

        :ivar bool normalize: If normalize is True, the function will return a normalized graph
            where each edge has been divided by the value of the largest edge.
        """

        # Initialize a new dictionary for our final weighted majority graph.
        wmgMap = dict()
        for cand in state:
            wmgMap[cand] = dict()
        for cand1, cand2 in itertools.combinations(state, 2):
            wmgMap[cand1][cand2] = 0
            wmgMap[cand2][cand1] = 0

        # Go through the wmgMaps and increment the value of each edge in our final graph with the
        # edges in each of the wmgMaps. We take into account the number of times that the vote
        # occured.
        for i in range(0, len(prefcounts)):
            # print("wmgMap=",wmgMap)
            for cand1, cand2 in itertools.combinations(rankmaps[i].keys(), 2):  # --------------------------
                # print("cand1=",cand1,"cand2=",cand2)
                # print(rankmaps[0][cand1] , rankmaps[0][cand2])
                if rankmaps[i][cand1] < rankmaps[i][cand2]:
                    wmgMap[cand1][cand2] += prefcounts[i]
                elif rankmaps[i][cand1] > rankmaps[i][cand2]:
                    wmgMap[cand2][cand1] += prefcounts[i]
                    # print("wmgMap=", wmgMap)
        # By default, we assume that the weighted majority graph should not be normalized. If
        # desired, we normalize by dividing each edge by the value of the largest edge.
        if normalize == True:
            maxEdge = float('-inf')
            for cand in wmgMap.keys():
                maxEdge = max(maxEdge, max(wmgMap[cand].values()))
            for cand1 in wmgMap.keys():
                for cand2 in wmgMap[cand1].keys():
                    wmgMap[cand1][cand2] = float(wmgMap[cand1][cand2]) / maxEdge

        print("wmg=", wmgMap)
        return wmgMap


class MechanismCoombs():
    def coombs_winners(self, profile):
        """
        Returns an integer list that represents all possible winners of a profile under Coombs rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        elecType = profile.getElecType()
        if elecType == "soc" or elecType == "csv":
            return self.coombssoc_winners(profile)
        elif elecType == "toc":
            return self.coombstoc_winners(profile)
        else:
            print("ERROR: unsupported profile type")
            exit()

    def coombssoc_winners(self, profile):
        """
        Returns an integer list that represents all possible winners of a profile under Coombs rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        ordering = profile.getOrderVectors()
        m = profile.numCands
        prefcounts = profile.getPreferenceCounts()
        if min(ordering[0]) == 0:
            startstate = set(range(m))
        else:
            startstate = set(range(1, m + 1))
        known_winners = set()
        # half = math.floor(n / 2.0)
        # ----------Some statistics--------------
        hashtable2 = set()

        # push the node of start state into the priority queue
        root = Node(value=startstate)
        stackNode = []
        stackNode.append(root)

        while stackNode:
            # ------------pop the current node----------------
            node = stackNode.pop()
            # -------------------------------------------------
            state = node.value.copy()
            # use heuristic to delete all the candidates which satisfy the following condition

            # goal state 1: if the state set contains only 1 candidate, then stop
            if len(state) == 1 and list(state)[0] not in known_winners:
                known_winners.add(list(state)[0])
                continue
            # goal state 2 (pruning): if the state set is subset of the known_winners set, then stop
            if state <= known_winners:
                continue
            # ----------Compute plurality score for the current remaining candidates-------------
            reverse_veto_score = self.get_reverse_veto_scores(prefcounts, ordering, state, m)

            # if current state satisfies one of the 3 goal state, continue to the next loop

            # After using heuristics, generate children and push them into priority queue
            # frontier = [val for val in known_winners if val in state] + list(set(state) - set(known_winners))

            maxscore = max(reverse_veto_score.values())
            for to_be_deleted in state:
                if reverse_veto_score[to_be_deleted] == maxscore:
                    child_state = state.copy()
                    child_state.remove(to_be_deleted)
                    tpc = tuple(sorted(child_state))
                    if tpc in hashtable2:
                        continue
                    else:
                        hashtable2.add(tpc)
                        child_node = Node(value=child_state)
                        stackNode.append(child_node)
        return sorted(known_winners)

    def get_reverse_veto_scores(self, prefcounts, ordering, state, m):
        plural_score = {}
        plural_score = plural_score.fromkeys(state, 0)
        for i in range(len(prefcounts)):
            for j in range(m - 1, -1, -1):
                if ordering[i][j] in state:
                    plural_score[ordering[i][j]] += prefcounts[i]
                    break
        return plural_score

    def coombstoc_winners(self, profile):
        """
        Returns an integer list that represents all possible winners of a profile under Coombs rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        ordering = profile.getOrderVectors()
        m = profile.numCands
        prefcounts = profile.getPreferenceCounts()
        rankmaps = profile.getRankMaps()
        if min(ordering[0]) == 0:
            startstate = set(range(m))
        else:
            startstate = set(range(1, m + 1))
        known_winners = set()
        # half = math.floor(n / 2.0)
        # ----------Some statistics--------------
        hashtable2 = set()

        # push the node of start state into the priority queue
        root = Node(value=startstate)
        stackNode = []
        stackNode.append(root)

        while stackNode:
            # ------------pop the current node----------------
            node = stackNode.pop()
            # -------------------------------------------------
            state = node.value.copy()
            # use heuristic to delete all the candidates which satisfy the following condition

            # goal state 1: if the state set contains only 1 candidate, then stop
            if len(state) == 1 and list(state)[0] not in known_winners:
                known_winners.add(list(state)[0])
                continue
            # goal state 2 (pruning): if the state set is subset of the known_winners set, then stop
            if state <= known_winners:
                continue
            # ----------Compute plurality score for the current remaining candidates-------------
            reverse_veto_score = self.get_reverse_veto_scores2(prefcounts, rankmaps, state)
            # print("reverse_veto_score = ",reverse_veto_score)

            # if current state satisfies one of the 3 goal state, continue to the next loop

            # After using heuristics, generate children and push them into priority queue
            # frontier = [val for val in known_winners if val in state] + list(set(state) - set(known_winners))

            maxscore = max(reverse_veto_score.values())
            for to_be_deleted in state:
                if reverse_veto_score[to_be_deleted] == maxscore:
                    child_state = state.copy()
                    child_state.remove(to_be_deleted)
                    tpc = tuple(sorted(child_state))
                    if tpc in hashtable2:
                        continue
                    else:
                        hashtable2.add(tpc)
                        child_node = Node(value=child_state)
                        stackNode.append(child_node)
        return sorted(known_winners)

    def get_reverse_veto_scores2(self, prefcounts, rankmaps, state):
        plural_score = {}
        plural_score = plural_score.fromkeys(state, 0)
        for i in range(len(prefcounts)):
            temp = list(filter(lambda x: x[0] in state, list(rankmaps[i].items())))
            max_value = max([value for key, value in temp])
            for j in state:
                if rankmaps[i][j] == max_value:
                    plural_score[j] += prefcounts[i]

        return plural_score


class MechanismRankedPairs():
    """
    The Ranked Pairs mechanism.
    This is the latest version created by Tyler Shepherd on
    Nov. 3 2018, originally in two_loop_LP_sampling_LP_11_3.py.
    """

    # debug_mode
    # = 0: no output
    # = 1: outputs only initial state
    # = 2: outputs on stop conditions
    # = 3: outputs all data
    def __init__(self):
        global debug_mode, BEGIN
        self.debug_mode = 0
        self.BEGIN = time.perf_counter()

        # Timeout in seconds
        self.TIMEOUT = 60 * 60 * 60

        self.tau_for_testing = 0.05

    class Stats:
        # Stores statistics being measured and updated throughout procedure
        """
        Stopping Conditions:
            1: G U E is acyclic
            2: possible_winners <= known_winners (pruning)
            3: exactly 1 cand with in degree 0
            4: G U Tier is acyclic (in max children method)
        """
        def __init__(self):
            self.discovery_states = dict()
            self.discovery_times = dict()
            self.num_nodes = 0
            self.num_outer_nodes = 0
            self.stop_condition_hits = {1: 0, 2: 0, 3: 0, 4: 0}
            self.num_hashes = 0
            self.num_initial_bridges = 0
            self.num_redundant_edges = 0
            self.num_sampled = 0
            self.sampled = []

    def output_graph(self, G):
        # Draws the given graph G using networkx

        pos = nx.circular_layout(G)  # positions for all nodes
        pos = dict(zip(sorted(pos.keys()), pos.values()))
        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=350)

        # edges
        nx.draw_networkx_edges(G, pos, width=3, alpha=0.5, edge_color='b')

        # labels
        nx.draw_networkx_labels(G, pos, font_size=14, font_family='sans-serif')

        plt.axis('off')
        plt.savefig("weighted_graph.png")  # save as png
        plt.show()  # display

    def add_winners(self, G, I, known_winners, stats, possible_winners = None):
        """
        Adds the winners of completed RP graph G
        :param G: networkx graph, should be final resulting graph after running RP
        :param I: list of all nodes
        :param known_winners: list of winners found so far, will be updated
        :param stats: Stats class storing run statistics
        :param possible_winners: Can optionally pass in possible winners if already computed to avoid re-computing here
        """
        if possible_winners is None:
            G_in_degree = G.in_degree(I)
            to_be_added = set([x[0] for x in G_in_degree if x[1] == 0])
        else:
            to_be_added = possible_winners
        for c in to_be_added:
            if c not in known_winners:
                known_winners.add(c)
                stats.discovery_states[c] = stats.num_nodes
                stats.discovery_times[c] = time.perf_counter() - self.BEGIN
                if self.debug_mode >= 2:
                    print("Found new winner:", c)

    def stop_conditions(self, G, E, I, known_winners, stats):
        """
        Determines if G, E state can be ended early
        :param G: networkx DiGraph of the current representation of "locked in" edges in RP
        :param E: networkx DiGraph of the remaining edges not yet considered
        :param I: list of all nodes
        :param known_winners: list of currently known PUT-winners
        :param stats: Stats object containing runtime statistics
        :return: -1 if no stop condition met, otherwise returns the int of the stop condition
        """

        in_deg = G.in_degree(I)
        possible_winners = [x[0] for x in in_deg if x[1] == 0]

        # Stop Condition 2: Pruning. Possible winners are subset of known winners
        if set(possible_winners) <= known_winners:
            stats.stop_condition_hits[2] += 1
            if self.debug_mode >= 2:
                print("Stop Condition 2: pruned")
            return 2

        # Stop Condition 3: Exactly one node has indegree 0
        if len(possible_winners) == 1:
            stats.stop_condition_hits[3] += 1
            if self.debug_mode >= 2:
                print("Stop Condition 3: one cand in degree 0")
            self.add_winners(G, I, known_winners, stats, possible_winners)
            return 3

        # Stop Condition 1: G U E is acyclic
        temp_G = nx.compose(G, E)
        if nx.is_directed_acyclic_graph(temp_G) is True:
            stats.stop_condition_hits[1] += 1
            if self.debug_mode >= 2:
                print("Stop Condition 1: acyclic")
            self.add_winners(G, I, known_winners, stats)
            return 1

        return -1

    def getWinners(self, profile):
        """
        Returns 1. a list of all PUT-winners of profile under ranked pairs rule
        and 2. A Stats object of runtime statistics

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Initialize
        stats = self.Stats()

        wmg = profile.getWmg()
        known_winners = set()
        I = list(wmg.keys())

        G = nx.DiGraph()
        G.add_nodes_from(I)

        E = nx.DiGraph()
        E.add_nodes_from(I)
        for cand1, cand2 in itertools.permutations(wmg.keys(), 2):
            if wmg[cand1][cand2] > 0:
                E.add_edge(cand1, cand2, weight=wmg[cand1][cand2])

        # print(wmg)
        # self.output_graph(E)

        # Sample
        num_samples = 200
        for i in range(num_samples):
            self.sample(E, I, known_winners, stats)


        stats.num_sampled = len(known_winners)
        stats.sampled = sorted(known_winners.copy())

        # Start search

        # Each node contains (G, E)
        root = Node(value=(G, E))
        stackNode = []
        stackNode.append(root)

        hashtable = set()

        while stackNode:
            # Pop new node to explore
            node = stackNode.pop()
            (G, E) = node.value

            # Check hash
            hash_state = hash(str(G.edges()) + str(E.edges()))
            if hash_state in hashtable:
                stats.num_hashes += 1
                if self.debug_mode == 3:
                    print("hashed in outer hashtable")
                continue
            hashtable.add(hash_state)

            stats.num_outer_nodes += 1
            stats.num_nodes += 1

            if self.debug_mode == 3:
                print("Popped new node: ")
                print("G:", G.edges())
                print("E:", E.edges())

            # Flag for whether expanding the current tier required finding max children
            f_found_max_children = 0

            # Continue performing RP on this state as long as tie-breaking order doesn't matter
            while len(E.edges()) != 0:
                if self.stop_conditions(G, E, I, known_winners, stats) != -1:
                    # Stop condition hit
                    break

                (max_weight, max_edge) = max([(d['weight'], (u, v)) for (u, v, d) in E.edges(data=True)])
                ties = [d['weight'] for (u, v, d) in E.edges(data=True)].count(max_weight)

                if ties == 1:
                    # Tier only has one edge
                    if self.debug_mode == 3:
                        print("Only 1 edge in tier")

                    E.remove_edges_from([max_edge])
                    if nx.has_path(G, max_edge[1], max_edge[0]) is False:
                        G.add_edges_from([max_edge])

                else:
                    # This tier has multiple edges with same max weight.
                    tier = [(u, v) for (u, v, d) in E.edges(data=True) if d['weight'] == max_weight]
                    if self.debug_mode == 3:
                        print("Tier =", tier)

                    E.remove_edges_from(tier)

                    # Compute "bridge edges" which are not in any cycle
                    Gc = G.copy()
                    Gc.add_edges_from(tier)
                    scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if
                           len(g.edges()) != 0]
                    bridges = set(Gc.edges()) - set(itertools.chain(*scc))
                    G.add_edges_from(bridges)
                    tier = list(set(tier) - bridges)

                    G_tc = nx.transitive_closure(G)

                    # Remove "inconsistent edges" that cannot be added to G without causing cycle
                    reverse_G = nx.DiGraph.reverse(G_tc)
                    tier = list(set(tier) - set(reverse_G.edges()))

                    # Remove "redundant edges": if there is already path from e[0] to e[1], can immediately add e
                    redundant_edges = set()
                    for e in tier:
                        if G_tc.has_edge(e[0], e[1]):
                            redundant_edges.add(e)
                            G.add_edges_from([e])
                    stats.num_redundant_edges += len(redundant_edges)
                    tier = list(set(tier) - redundant_edges)

                    if len(tier) == 0:
                        # No need to find max children, as tier is now empty
                        continue

                    max_children = self.find_max_children_scc_decomposition(G, tier, scc, bridges, I, known_winners, stats)

                    # Determine priority ordering of maximal children
                    children = dict()
                    index = 0
                    for child in max_children:
                        # child_node = Node(value=(self.edges2string(child.edges(), I), self.edges2string(E.edges(), I)))
                        child_node = Node(value=(child, E.copy()))
                        c_in_deg = child.in_degree(I)
                        available = set([x[0] for x in c_in_deg if x[1] == 0])
                        priority = len(available - known_winners)
                        # children[child_node] = (priority, index)
                        children[child_node] = index
                        child.add_nodes_from(I)
                        index += 1
                        continue

                    children_items = sorted(children.items(), key=lambda x: x[1])
                    sorted_children = [key for key, value in children_items]
                    stackNode += sorted_children
                    f_found_max_children = 1
                    break

            # f_found_max_children is needed since, if we just added more nodes to stack, then current (G, E) is not actual valid state
            if len(E.edges()) == 0 and f_found_max_children == 0:
                # E is empty
                if self.debug_mode >= 2:
                    print("E is empty")
                self.add_winners(G, I, known_winners, stats)

        return sorted(known_winners), stats

    def edges2string(self, edges, I):
        m = len(I)
        gstring = list(str(0).zfill(m**2))
        for e in edges:
            gstring[(e[0] - min(I))*m + e[1] - min(I)] = '1'

        return ''.join(gstring)

    def string2edges(self, gstring, I):
        m = len(I)
        edges = []
        for i in range(len(gstring)):
            if gstring[i] == '1':
                e1 = i % m + min(I)
                e0 = int((i - e1) / m) + min(I)
                edges.append((e0, e1))
        return edges

    def find_max_children_scc_decomposition(self, G, tier, scc, bridges, I, known_winners, stats):
        '''
        Finds the maximal children of G when tier is added using SCC decomposition
        :param G: Networkx DiGraph of edges "locked in" so far
        :param tier: List of edges in the current tier to be added with equal weight
        :param scc: List of the strongly connected components of G U tier, each being a list of edges
        :param bridges: List of edges that are bridges between sccs of G U tier
        :param I: List of all nodes
        :param known_winners: Known PUT-winners computed by RP so far
        :param stats: Stats object containing runtime statistics
        :return: Array of Networkx DiGraphs that are the maximal children of G U T
        '''

        if len(scc) == 1:
            children = self.explore_max_children_lp(G, tier, I, known_winners, stats)
            return children

        mc_list = []

        for x in scc:
            G_temp = nx.DiGraph(list(set(G.edges()).intersection(set(x))))
            T_temp = list(set(tier).intersection(set(x)))
            temp = self.explore_max_children_lp(G_temp, T_temp, I, known_winners, stats, f_scc = 1)
            mc_list.append(temp)

        Cartesian = itertools.product(*mc_list)
        return [nx.DiGraph(list(set(itertools.chain(*[list(y.edges()) for y in x])).union(bridges))) for x in Cartesian]

    def explore_max_children_lp(self, G, tier, I, known_winners, stats, f_scc = 0):
        """
        Computes the maximal children of G when tier is added
        :param G: DiGraph, A directed graph
        :param tier: list of tuples which correspond to multiple edges with same max weight.
                    e.g. edges = [x for x in wmg2.keys() if wmg2[x] == max_weight]
        :param I: all nodes in G
        :param known_winners: PUT-winners found so far by RP
        :param stats: Stats object
        :param f_scc: set to 1 if the G and tier being considered are an SCC of the full graph due to SCC decomposition
        :return: set of graphs which correspond to maximum children of given parent: G
        """

        # self.output_graph(G)
        # self.output_graph(nx.DiGraph(tier))

        max_children = []
        cstack = []

        # print("start mc:", time.perf_counter() - self.BEGIN)

        hashtable = set()

        if self.debug_mode >= 1:
            print("Exploring max children")
            print("G:", G.edges())
            print("Tier:", tier)
            print("Known winners:", known_winners)
            print("---------------------------")

        in_deg = G.in_degree()
        nodes_with_no_incoming = set()
        for x in in_deg:
            if x[1] == 0:
                nodes_with_no_incoming.add(x[0])
        for x in I:
            if x not in G.nodes():
                nodes_with_no_incoming.add(x)

        root = Node(value=(self.edges2string(G.edges(), I), self.edges2string(tier, I), nodes_with_no_incoming))
        cstack.append(root)

        END = self.BEGIN + self.TIMEOUT

        while cstack:
            node = cstack.pop()
            (G_str, T_str, no_incoming) = node.value

            if time.perf_counter() > END:
                print("TIMEOUT")
                return max_children

            # Check hash. Doesn't ever happen if the below hash is included
            hash_G = hash(G_str)
            if hash_G in hashtable:
                stats.num_hashes += 1
                print('hash')
                if self.debug_mode >= 2:
                    print("hashed in hashtable")
                continue
            hashtable.add(hash_G)

            stats.num_nodes += 1

            G = nx.DiGraph(self.string2edges(G_str, I))
            T = self.string2edges(T_str, I)
            G.add_nodes_from(I)

            if self.debug_mode == 3:
                print("popped")
                print("G: ", G.edges())
                print("T: ", T)

            # goal state 2: if current G's possible winners is subset of known winners,
            # then directly ignore it.
            if no_incoming <= known_winners and not f_scc:
                stats.stop_condition_hits[2] += 1
                if self.debug_mode >= 3:
                    print("MC goal state 2: pruned")
                continue

            # goal state 1: if there are no edges to be added, then add the G_
            if len(T) == 0:
                max_children.append(G.copy())
                if self.debug_mode >= 2:
                    print("MC goal state 1: no more edges in tier")
                    print("max child: ", G.edges())
                continue

            # goal state 3: if current G has exactly one cand with in degree 0, it is a PUT-winner
            if len(no_incoming) == 1 and not f_scc:
                stats.stop_condition_hits[3] += 1
                if self.debug_mode >= 2:
                    print("MC goal state 3: only one cand in degree 0")
                    print("max child:", G.edges())
                self.add_winners(G, I, known_winners, stats, no_incoming)
                continue

            # goal state 4: if union of current G and edges is acyclic,
            # then directly add it to the max_children_set
            Gc = G.copy()
            Gc.add_edges_from(T)
            if nx.is_directed_acyclic_graph(Gc):
                stats.stop_condition_hits[4] += 1

                hash_temp_G = hash(self.edges2string(Gc.edges(), I))
                if hash_temp_G not in hashtable:
                    hashtable.add(hash_temp_G)
                    max_children.append(Gc)

                    if self.debug_mode >= 2:
                        print("MC goal state 4: G U T is acyclic")
                        print("max child:", Gc.edges())
                else:
                    stats.num_hashes += 1
                continue

            # Perform reductions every step:

            # Compute "bridge edges" which are not in any cycle
            Gc = G.copy()
            Gc.add_edges_from(T)
            scc = [list(g.edges()) for g in nx.strongly_connected_component_subgraphs(Gc, copy=True) if
                   len(g.edges()) != 0]
            bridges = set(Gc.edges()) - set(itertools.chain(*scc))
            G.add_edges_from(bridges)
            T = list(set(T) - bridges)

            G_tc = nx.transitive_closure(G)

            # Remove "inconsistent edges" that cannot be added to G without causing cycle
            reverse_G = nx.DiGraph.reverse(G_tc)
            T = list(set(T) - set(reverse_G.edges()))

            # Remove "redundant edges": if there is already path from e[0] to e[1], can immediately add e
            redundant_edges = set()
            for e in T:
                if G_tc.has_edge(e[0], e[1]):
                    redundant_edges.add(e)
                    G.add_edges_from([e])
            stats.num_redundant_edges += len(redundant_edges)
            T = list(set(T) - redundant_edges)

            # Flag for whether adding any edge from T causes G to remain acyclic
            f_isAcyclic = 0

            children = dict()

            # Used to break ties
            index = 0
            T = sorted(T)
            for e in T:
                G.add_edges_from([e])
                Gc_str = self.edges2string(G.edges(), I)
                if hash(Gc_str) in hashtable:
                    f_isAcyclic = 1

                    stats.num_hashes += 1
                    G.remove_edges_from([e])
                    continue

                if not nx.has_path(G, source=e[1], target=e[0]):
                    f_isAcyclic = 1

                    Tc = copy.deepcopy(T)
                    Tc.remove(e)

                    # Remove the head of the edge if it had no incoming edges previously
                    no_incoming_c = no_incoming.copy()
                    no_incoming_c.discard(e[1])

                    child = Node(value=(Gc_str, self.edges2string(Tc, I), no_incoming_c))

                    priority = len(no_incoming_c - known_winners)

                    children[child] = (priority, index)
                    index = index + 1

                    if self.debug_mode == 3:
                        print("add new child with edge ", e, " and priority ", priority)

                G.remove_edges_from([e])

            children_items = sorted(children.items(), key=lambda x: (x[1][0], x[1][1]))
            sorted_children = [key for key, value in children_items]
            cstack += sorted_children

            # goal state 5: adding all edges in T individually cause G to be cyclic
            if f_isAcyclic == 0:
                max_children.append(G.copy())

                if self.debug_mode >= 2:
                    print("MC goal state 5 - found max child")
                    print("max child: ", G.edges())
                continue

        if self.debug_mode >= 1:
            print("finished exploring max children")
            print("num max children:", len(max_children))
            print("PUT-winners:", known_winners)

        return max_children

    def sample(self, E, I, known_winners, stats):
        '''
        Using random tie-breaking, run through one procedure of RP and add resulting winner to known_winners
        :param E: DiGraph, All postive edges in the wmg
        :param I: List of all nodes in E
        :param known_winners: Set of already-discovered PUT-winners to be added to
        :param stats: Stats object storing runtime statistics
        :return: Nothing
        '''
        G = nx.DiGraph()
        G.add_nodes_from(I)

        Ec = E.copy()

        while len(Ec.edges()) != 0:
            max_weight = max([(d['weight']) for (u, v, d) in Ec.edges(data=True)])
            tier = [(u, v) for (u, v, d) in Ec.edges(data=True) if d['weight'] == max_weight]

            # e = tier[random.randint(0, len(tier) -1 )]
            priorities = []
            potential_winners = set([x[0] for x in G.in_degree(I) if x[1] == 0])
            base_priority = len(potential_winners - known_winners)
            for e in tier:
                if G.in_degree(e[1]) == 0 and e[1] not in known_winners:
                    priority = base_priority - 1
                else:
                    priority = base_priority
                priorities.append(exp(priority / self.tau_for_testing))
            q_sum = sum(priorities)
            probs = []
            for v in priorities:
                probs.append(v / q_sum)
            legal_actions_index = [i for i in range(len(tier))]
            e = tier[np.random.choice(legal_actions_index, p=probs)]

            if not nx.has_path(G, e[1], e[0]):
                G.add_edges_from([e])
                Ec.remove_edges_from([e])
            else:
                Ec.remove_edges_from([e])

        self.add_winners(G, I, known_winners, stats)


class MechanismBlack():
    """
    The Black mechanism.
    """

    def black_winner(self, profile):
        """
        Returns a number or a list that associates the winner(s) of a profile under black rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are
        # allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        wmg = profile.getWmg()
        m = profile.numCands
        for cand1 in wmg.keys():
            outgoing = 0
            for cand2 in wmg[cand1].keys():
                if wmg[cand1][cand2] > 0:
                    outgoing += 1
            if outgoing == m - 1:
                return [cand1]

        Borda_winner = MechanismBorda().getWinners(profile)
        return Borda_winner


class MechanismPluralityRunOff():
    """
    The Plurality with Runoff mechanism.
    """

    def PluRunOff_single_winner(self, profile):
        """
        Returns a number that associates the winner of a profile under Plurality with Runoff rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are
        # allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        # Initialization
        prefcounts = profile.getPreferenceCounts()
        len_prefcounts = len(prefcounts)
        rankmaps = profile.getRankMaps()
        ranking = MechanismPlurality().getRanking(profile)

        # 1st round: find the top 2 candidates in plurality scores
        # Compute the 1st-place candidate in plurality scores
        print(ranking)
        max_cand = ranking[0][0][0]

        # Compute the 2nd-place candidate in plurality scores
        # Automatically using tie-breaking rule--numerically increasing order
        if len(ranking[0][0]) > 1:
            second_max_cand = ranking[0][0][1]
        else:
            second_max_cand = ranking[0][1][0]

        top_2 = [max_cand, second_max_cand]
        # 2nd round: find the candidate with maximum plurality score
        dict_top2 = {max_cand: 0, second_max_cand: 0}
        for i in range(len_prefcounts):
            vote_top2 = {key: value for key, value in rankmaps[i].items() if key in top_2}
            top_position = min(vote_top2.values())
            keys = [x for x in vote_top2.keys() if vote_top2[x] == top_position]
            for key in keys:
                dict_top2[key] += prefcounts[i]

        # print(dict_top2)
        winner = max(dict_top2.items(), key=lambda x: x[1])[0]

        return winner

    def PluRunOff_cowinners(self, profile):
        """
        Returns a list that associates all the winners of a profile under Plurality with Runoff rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are
        # allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        # Initialization
        prefcounts = profile.getPreferenceCounts()
        len_prefcounts = len(prefcounts)
        rankmaps = profile.getRankMaps()
        ranking = MechanismPlurality().getRanking(profile)

        known_winners = set()
        # 1st round: find the top 2 candidates in plurality scores
        top_2_combinations = []
        if len(ranking[0][0]) > 1:
            for cand1, cand2 in itertools.combinations(ranking[0][0], 2):
                top_2_combinations.append([cand1, cand2])
        else:
            max_cand = ranking[0][0][0]
            if len(ranking[0][1]) > 1:
                for second_max_cand in ranking[0][1]:
                    top_2_combinations.append([max_cand, second_max_cand])
            else:
                second_max_cand = ranking[0][1][0]
                top_2_combinations.append([max_cand, second_max_cand])

        # 2nd round: find the candidate with maximum plurality score
        for top_2 in top_2_combinations:
            dict_top2 = {top_2[0]: 0, top_2[1]: 0}
            for i in range(len_prefcounts):
                vote_top2 = {key: value for key, value in rankmaps[i].items() if key in top_2}
                top_position = min(vote_top2.values())
                keys = [x for x in vote_top2.keys() if vote_top2[x] == top_position]
                for key in keys:
                    dict_top2[key] += prefcounts[i]

            max_value = max(dict_top2.values())
            winners = [y for y in dict_top2.keys() if dict_top2[y] == max_value]
            known_winners = known_winners | set(winners)

        return sorted(known_winners)

    def getMov(self, profile):
        """
        Returns an integer that is equal to the margin of victory of the election profile.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        # from . import mov
        import mov
        return mov.MoVPluRunOff(profile)


"""
Multi-winner voting rules
"""


class MechanismSNTV():
    """
    The Single non-transferable vote mechanism.
    """

    def SNTV_winners(self, profile, K):
        """
        Returns a list that associates all the winners of a profile under Single non-transferable vote rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are
        # allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        m = profile.numCands
        candScoresMap = MechanismPlurality().getCandScoresMap(profile)
        if K >= m:
            return list(candScoresMap.keys())
        # print(candScoresMap)
        sorted_items = sorted(candScoresMap.items(), key=lambda x: x[1], reverse=True)
        sorted_dict = {key: value for key, value in sorted_items}
        winners = list(sorted_dict.keys())[0:K]
        return winners

    def getMov(self, profile, K):
        """
        Returns an integer that is equal to the margin of victory of the election profile.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        # from . import mov
        import mov
        return mov.MoV_SNTV(profile, K)


class MechanismChamberlin_Courant():
    """
    The Chamberlin–Courant mechanism.
    """

    def single_peaked_winners(self, profile, d=1, K=3, funcType='Borda', scoringVector=[]):
        """
        Returns a list that associates all the winners of a profile under The Chamberlin–Courant rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # Currently, we expect the profile to contain complete ordering over candidates. Ties are
        # allowed however.
        elecType = profile.getElecType()
        if elecType != "soc" and elecType != "toc" and elecType != "csv":
            print("ERROR: unsupported election type")
            exit()

        # ------------------1. INITIALIZATION-----------------------------
        m = profile.numCands
        n = profile.numVoters
        cand = list(profile.candMap.keys())
        cand.append(cand[m - 1] + 1)
        theta = n - d
        if funcType == 'Borda':
            scoringVector = MechanismBorda().getScoringVector(profile)
        z = dict()
        for k in range(1, K + 2):  # k = 1,...,K + 1
            z[k] = dict()
            for j in range(1, m + 2):
                z[k][j] = dict()

        for j in range(1, m + 2):
            for t in range(0, theta + 1):
                z[1][j][t] = self.s(profile, 1, j, t, {cand[j - 1]}, scoringVector)
                for k in range(1, K + 1):
                    z[k + 1][j][t] = float("-inf")

        # ------------------2. MAIN LOOP-----------------------------
        for k in range(1, K + 1):
            # Predecessors loop:
            for p in range(1, m + 1):
                for u in range(0, theta + 1):
                    if z[k][p][u] != float("-inf"):
                        # Successors sub-loop:
                        for j in range(p + 1, m + 2):
                            for t in range(u, theta + 1):
                                z[k + 1][j][t] = max(z[k + 1][j][t], z[k][p][u]
                                                     + self.s(profile, p + 1, j, t - u, {cand[p - 1], cand[j - 1]},
                                                              scoringVector))

        max_utility = z[K + 1][m + 1][theta]
        print("max_utility=", max_utility)
        # --------------------3. OUTPUT WINNERS---------------------------
        winners = []
        temp_max = max_utility
        j = m + 1
        t = theta
        for k in range(K + 1, 1, -1):
            z_k_j_t = array(
                [[z[k - 1][p][u] + self.s(profile, p + 1, j, t - u, {cand[p - 1], cand[j - 1]}, scoringVector)
                  for u in range(0, theta + 1)] for p in range(1, m + 1)])
            p_ind = where(temp_max == z_k_j_t)[0][0]
            u_ind = where(temp_max == z_k_j_t)[0][0]
            p0 = list(range(1, m + 1))[p_ind]
            u0 = list(range(0, theta + 1))[u_ind]
            winners.append(p0)
            temp_max = z[k][p0][u0]
            j = p0
            t = u0

        return sorted(winners)

    def s(self, profile, l, j, t, S, scoringVector):
        new_prefcounts, new_rankmaps = self.V(profile, l, j)
        # print(new_prefcounts, new_rankmaps)
        if t == 0 or len(new_prefcounts) == 0:
            return float("-inf")

        s_S = []
        for i in range(len(new_prefcounts)):
            s_S.append(max(scoringVector[new_rankmaps[i][x] - 1]
                           if x in new_rankmaps[i].keys() else float("-inf") for x in S))

        ind = (-array(s_S)).argsort()
        return dot(array(s_S)[ind][0:t], array(new_prefcounts)[ind][0:t])

    def V(self, profile, l, j):
        prefcounts = profile.getPreferenceCounts()
        rankmaps = profile.getRankMaps()
        cand = list(profile.candMap.keys())
        m = len(cand)
        cand.append(cand[m - 1] + 1)
        new_prefcounts = []
        new_rankmaps = []
        for i in range(len(prefcounts)):
            top_i = list(rankmaps[i].keys())[list(rankmaps[i].values()).index(1)]
            if top_i in range(cand[l - 1], cand[j - 1] + 1):
                new_prefcounts.append(prefcounts[i])
                new_rankmaps.append(rankmaps[i])
        return new_prefcounts, new_rankmaps


class MechanismBordaMean():
    """
        The Borda-mean mechanism.
        """

    def Borda_mean_winners(self, profile):
        """
        Returns a list that associates all the winners of a profile under The Borda-mean rule.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        n_candidates = profile.numCands
        prefcounts = profile.getPreferenceCounts()
        len_prefcounts = len(prefcounts)
        rankmaps = profile.getRankMaps()
        values = zeros([len_prefcounts, n_candidates], dtype=int)
        if min(list(rankmaps[0].keys())) == 0:
            delta = 0
        else:
            delta = 1
        for i in range(len_prefcounts):
            for j in range(delta, n_candidates + delta):
                values[i][j - delta] = rankmaps[i][j]
        # print("values=", values)
        mat0 = self._build_mat(values, n_candidates, prefcounts)
        borda = [0 for i in range(n_candidates)]
        for i in range(n_candidates):
            borda[i] = sum([mat0[i, j] for j in range(n_candidates)])
        borda_mean = mean(borda)
        bin_winners_list = [int(borda[i] >= borda_mean) for i in range(n_candidates)]
        return bin_winners_list

    def _build_mat(self, ranks, n_candidates, prefcounts):
        """
        Builds  mxm matrix. Entry at i,j has #i>j - #i<j
        :param ranks:
        :return: mxm matrix
        """

        mat = zeros((n_candidates, n_candidates))
        for i, j in itertools.combinations(range(n_candidates), 2):
            preference = ranks[:, i] - ranks[:, j]
            h_ij = dot((preference < 0), prefcounts)  # prefers i to j
            h_ji = dot((preference > 0), prefcounts)  # prefers j to i
            mat[i, j] = h_ij - h_ji
            mat[j, i] = h_ji - h_ij
        return mat

    """
    Simulate approval voting for any k-chotomous preferences where voters can have different values of k.
    Borda-mean rule is applied to each vote, to change any k-chotomous preferences to dichotomous approval votes. 
    Each vote is simulated as an approval vote.
    The approval rule is then used to aggregate these simulated approval votes.
    Input:
        ranks: a nxm matrix. rows are for votes, columns for candidates. i,j-th entry gives rank of candidate j by voter i.
            only the relative numbers/positions matter.
    Output:
        winners: an m-dimensional array. j-th entry is 1 if candidate j is a winner by the approval voting rule.
        approval_score: an m-dimenstional array: j-th entry gives approval score of candidate j.
    """

    def simulated_approval(self, profile):
        n_candidates = profile.numCands
        n_voters = profile.numVoters
        prefcounts = profile.getPreferenceCounts()
        len_prefcounts = len(prefcounts)
        rankmaps = profile.getRankMaps()
        values = zeros([len_prefcounts, n_candidates], dtype=int)
        if min(list(rankmaps[0].keys())) == 0:
            delta = 0
        else:
            delta = 1
        for i in range(len_prefcounts):
            for j in range(delta, n_candidates + delta):
                values[i][j - delta] = rankmaps[i][j]
        approval = list()
        for i in range(n_voters):
            vote = array([list(values[i, :])])
            approvals = self.borda_mean(vote)
            approval.append(approvals)
        return self.approval_rule(array(approval))

    """
    Compute approval rule
    Input:
        approval: an nxm matrix of approval votes. i,j-th entry is 1 if voter i approves candidate j; 0 otherwise.
    Output:
        winners: an m-dimensional array. j-th entry is 1 if candidate j is a winner by the approval voting rule.
        approval_score: an m-dimenstional array: j-th entry gives approval score of candidate j.    
    """

    def approval_rule(self, approval):
        n_voters, n_candidates = approval.shape
        approval_score = [0 for j in range(n_candidates)]
        for i in range(n_voters):
            approvals = approval[i, :]
            approval_score = [approval_score[j] + approvals[j] for j in range(n_candidates)]
        max_score = max(approval_score)
        winners = list((array(approval_score) >= max_score).astype(int))
        return winners, approval_score

    """
    Build weighted tournament graph from any k-chotomous preferences. Different votes can have different values of k.
    Input:
        ranks: a nxm matrix. rows are for votes, columns for candidates. i,j-th entry gives rank of candidate j by voter i.
            only the relative numbers/positions matter.
    Output:
        mat: a mxm matrix. i,j-th entry gives |i>j| - |j>i|, ties are ignored, can have -ve entries.
    """

    def _build_mat_app(self, ranks):
        n_voters, n_candidates = ranks.shape
        mat = zeros((n_candidates, n_candidates))
        for i, j in itertools.combinations(range(n_candidates), 2):
            preference = ranks[:, i] - ranks[:, j]
            h_ij = sum(preference < 0)  # prefers i to j
            h_ji = sum(preference > 0)  # prefers j to i
            mat[i, j] = h_ij - h_ji
            mat[j, i] = h_ji - h_ij
        return mat

    """
    Compute the Borda mean rule.
    Input:
        ranks: a nxm matrix. rows are for votes, columns for candidates. i,j-th entry gives rank of candidate j by voter i.
            only the relative numbers/positions matter.
    Output:
        winners: an m-dimensional array. j-th entry is 1 if candidate j is a winner by the Borda mean rule, 0 otherwise.
        borda: m-deimensional array with Borda mean scores, can be used to generate a ranking. 
            Sum of edges from candidate j to every other candidate.
    """

    def borda_mean(self, ranks):
        mat = self._build_mat_app(ranks)
        n_voters, n_candidates = ranks.shape
        borda = [0 for i in range(n_candidates)]
        for i in range(n_candidates):
            borda[i] = sum([mat[i, j] for j in range(n_candidates)])
        borda_mean = mean(borda)
        winners = [int(borda[i] >= borda_mean) for i in range(n_candidates)]
        return winners


class Node:
    def __init__(self, value=None):
        self.value = value

    def __lt__(self, other):
        return 0

    def getvalue(self):
        return self.value

