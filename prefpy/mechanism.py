"""
Author: Kevin J. Hwang
"""
import io
import math

import itertools
from .preference import Preference
from .profile import Profile

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
                candScoresMap[cand] += scoringVector[rankMap[cand]-1]*rankMapCount
        
        return candScoresMap

    def getMov(self, profile):
        """
        Returns an integer that is equal to the margin of victory of the election profile, that is,
        the number of votes needed to be changed to change to outcome into a draw.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        from . import mov
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
        
    def getCandScoresMap(self, profile):
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
        
        for i in range(0, len(rankMaps)):
            scoringVector  = []
            rankMap = rankMaps[i]
            rankMapCount = rankMapCounts[i]
            x = max(rankMap.values())
            for y in range(0,x-1):
                scoringVector.append(1)
            scoringVector.append(0)
            for cand in rankMap.keys():
                candScoresMap[cand] += scoringVector[rankMap[cand]-1]*rankMapCount
        print(candScoresMap)
        return candScoresMap

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
        score = profile.numCands-1
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
            for t in range(1, profile.numCands+1):
                for i in range(0, len(rankMaps)):        
                    if (rankMaps[i][cand] == t):
                        numTimesRanked += preferenceCounts[i]
                if numTimesRanked >= math.ceil(float(profile.numVoters)/2):
                    bucklinScores[cand] = t
                    break

        return bucklinScores

    def getMov(self, profile):
        """
        Returns an integer that is equal to the margin of victory of the election profile, that is,
        the number of votes needed to be changed to change to outcome into a draw.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        from . import mov
        return mov.movSimplifiedBucklin(profile)

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
            
                #If a pair of candidates is tied, we add alpha to their score for each vote.
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

        for i in range(1, numCands+1):
            for j in range(1, numCands+1):
                if (i == j):
                    continue
                if pairwisePreferences[i][j] > pairwisePreferences[j][i]:
                    strongestPaths[i][j] = pairwisePreferences[i][j]
                else:
                    strongestPaths[i][j] = 0

        for i in range(1, numCands+1):
            for j in range(1, numCands+1):
                if (i == j):
                    continue
                for k in range(1, numCands+1):
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

    #runs through list1
    for itr1 in range(0, len(list1) - 1):
        #runs through list2
        for itr2 in range(itr1 + 1, len(list2)):
            # checks if there is a discrepancy. If so, adds
            if ((list1[itr1] > list1[itr2]
                and list2[itr1] < list2[itr2])
            or (list1[itr1] < list1[itr2]
                and list2[itr1] > list2[itr2])):

                kt += 1
    # normalizes between 0 and 1
    kt = (kt * 2) / (len(list1) * (len(list1) - 1))

    #returns found value
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
        Returns an integer list that represents all winners of a profile.
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
        Returns an integer list that represents all winners of a profile.
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
        # values = zeros([len_prefcounts, m], dtype=int)
        #
        # for i in range(len_prefcounts):
        #     values[i] = list(rankmaps[i].values())
        #
        # # print values
        # dict0 = {}
        # for t in startstate:
        #     dict0[t] = values[:, t - 1]
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
        Returns an integer list that represents all winners of a profile.
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
        Returns an integer list that represents all winners of a profile.
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
            for cand1, cand2 in itertools.combinations(rankmaps[i].keys(), 2):  # --------------------------
                if rankmaps[0][cand1] < rankmaps[0][cand2]:
                    wmgMap[cand1][cand2] += prefcounts[i]
                elif rankmaps[0][cand1] > rankmaps[0][cand2]:
                    wmgMap[cand2][cand1] += prefcounts[i]

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

class MechanismCoombs():
    """
    The Baldwin mechanism.
    """
    def coombs_winners(self, profile):
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
        Returns an integer list that represents all winners of a profile.
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
        Returns an integer list that represents all winners of a profile.
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



class Node:
    def __init__(self, value=None):
        self.value = value

    def __lt__(self, other):
        return 0

    def getvalue(self):
        return self.value

