"""
Author: Kevin J. Hwang
"""

from . import prefpy_io
from numpy import *
import math
import itertools
# from .preference import Preference
from .profile import Profile
from .mechanism import *
# from mov import *
from .preference import Preference

# def movPosScoring(profile, scoringVector):
#     """
#     Returns an integer that is equal to the margin of victory of a profile, that is, the number of
#     votes needed to be changed to change to winner when using the positional scoring rule.
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     :ivar list<int> scoringVector: A list of integers (or floats) that give the scores assigned to
#         each position in a ranking from first to last.
#     """
#
#     # Currently, we expect the profile to contain complete ordering over candidates.
#     elecType = profile.getElecType()
#     if elecType != "soc" and elecType != "toc":
#         print("ERROR: unsupported election type")
#         exit()
#
#     # If the profile already results in a tie, return 0.
#     from . import mechanism
#     posScoring = mechanism.MechanismPosScoring(scoringVector)
#     winners = posScoring.getWinners(profile)
#     if len(winners) > 1:
#         return 1
#
#     rankMaps = profile.getRankMaps()
#     preferenceCounts = profile.getPreferenceCounts()
#     winner = winners[0]
#     candScoresMap = posScoring.getCandScoresMap(profile)
#     mov = profile.numVoters
#     # For each candidate, calculate the difference in scores that changing a vote can do.
#     for cand in profile.candMap.keys():
#         scoreEffects = []
#         if cand == winner: continue
#         for i in range(0, len(rankMaps)):
#             rankMap = rankMaps[i]
#             incInCandScore = scoringVector[0]-scoringVector[rankMap[cand]-1]
#             decInWinnerScore = scoringVector[rankMap[winner]-1]-scoringVector[-1]
#
#             # Create a tuple that contains the max increase/decrease and the rankmap count.
#             scoreEffect = (incInCandScore, decInWinnerScore, preferenceCounts[i])
#             scoreEffects.append(scoreEffect)
#
#         scoreEffects = sorted(scoreEffects, key=lambda scoreEffect: scoreEffect[0] + scoreEffect[1], reverse = True)
#
#         # We simulate the effects of changing the votes starting with the votes that will have the
#         # greatest impact.
#         winnerScore = candScoresMap[winner]
#         candScore = candScoresMap[cand]
#         votesNeeded = 0
#
#         for i in range(0, len(scoreEffects)):
#             scoreEffect = scoreEffects[i]
#             ttlChange = scoreEffect[0] + scoreEffect[1]
#
#
#             # Check if changing all instances of the current vote can change the winner.
#             if (ttlChange*scoreEffect[2] >= winnerScore-candScore):
#                 votesNeeded += math.ceil(float(winnerScore-candScore)/float(ttlChange))+1
#                 break
#
#             # Otherwise, update the election simulation with the effects of the current votes.
#             else:
#                 votesNeeded += scoreEffect[2]
#                 winnerScore -= scoreEffect[1]*scoreEffect[2]
#                 candScore += scoreEffect[0]*scoreEffect[2]
#
#             # If the number of votes needed to make the current candidate the winner is greater than
#             # the lowest number of votes needed to make some candidate the winner, we can stop
#             # trying.
#             if votesNeeded > mov:
#                 break
#
#         mov = min(mov,votesNeeded)
#
#     return int(mov)
#
#
# def movPlurality(profile):
#     """
#     Returns an integer that is equal to the margin of victory, that is, the number of votes needed
#     to be changed to change the winner when using the plurality rule.
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     """
#
#     scoringVector = []
#     scoringVector.append(1)
#     for i in range(1, profile.numCands):
#         scoringVector.append(0)
#     #return movPosScoring(profile, scoringVector)
#     return -1
#
# def movVeto(profile):
#     """
#     Returns an integer that is equal to the margin of victory, that is, the number of votes needed
#     to be changed to change the winner when using the veto rule.
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     """
#
#     scoringVector = []
#     for i in range(0, profile.numCands-1):
#         scoringVector.append(1)
#     scoringVector.append(0)
#     return movPosScoring(profile, scoringVector)
#
# def movBorda(profile):
#     """
#     Returns an integer that is equal to the margin of victory, that is, the number of votes needed
#     to be changed to change the winner when using the veto rule.
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     """
#
#     scoringVector = []
#     score = profile.numCands-1
#     for i in range(0, profile.numCands):
#         scoringVector.append(score)
#         score -= 1
#     return movPosScoring(profile, scoringVector)
#
#
# def movKApproval(profile, k):
#     """
#     Returns an integer that is equal to the margin of victory, that is, the number of votes needed
#     to be changed to change the winner when using the k-approval rule.
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     :ivar int k: A value for k.
#     """
#     scoringVector = []
#     for i in range(0, k):
#         scoringVector.append(1)
#     for i in range(k, profile.numCands):
#         scoringVector.append(0)
#     return movPosScoring(profile, scoringVector)


def MoVScoring(profile, scoringVector):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    :ivar list<int> scoringVector: A list of integers (or floats) that give the scores assigned to
        each position in a ranking from first to last.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    n = profile.numVoters
    m = profile.numCands
    if len(scoringVector) != m:
        print("ERROR: the length of the scoring vector is not correct!")
        exit()

    # Construct the score matrix--values
    prefcounts = array(profile.getPreferenceCounts())
    rankmaps = profile.getRankMaps()
    len_prefcounts = len(prefcounts)
    values = zeros([len_prefcounts, m], dtype=int)
    ranks = zeros([len_prefcounts, m], dtype=int)
    for i in range(len_prefcounts):
        ranks[i] = list(rankmaps[i].values())
        for j in range(len(ranks[i])):
            values[i][j] = scoringVector[ranks[i][j]-1]

    # Compute the scores of all the candidates
    score = dot(array(prefcounts), values)
    # Compute the winner of the original profile
    d = argmax(score, axis=0) + 1
    alter = delete(range(1, m + 1), d - 1)
    # Initialize
    MoV = n * ones(m, dtype=int)
    for c in alter:
        # The difference vector of d and c
        difference = values[:, c - 1] - values[:, d - 1]
        index = argsort(difference, axis=0, kind='mergesort')

        # The vector that each element is the gain in the difference
        # between d and c if the pattern of the vote changed to [c > others > d]
        change = m - 1 - difference

        # The total_difference between score(d) and score(c)
        total_difference = score[d - 1] - score[c - 1]
        for i in range(len_prefcounts):
            # The number of votes of the first i kinds of patterns
            temp_sum = sum(prefcounts[index][0:i])

            # The aggregate gain (of the first i kinds of patterns)
            # in the difference between d and c if changed to [c > others > d]
            lower_bound = dot(prefcounts[index][0:i], change[index][0:i])

            # The aggregate gain (of the first i+1 kinds of patterns)
            # in the difference between d and c if changed to [c > others > d]
            upper_bound = dot(prefcounts[index][0:i + 1], change[index][0:i + 1])
            if lower_bound < total_difference <= upper_bound:
                # tie-breaking rule: numerically increasing order
                # if c > d, score(c) = score (d), the winner is still d,
                # so only when score(c) is strictly less than score(d), will the winner change to c.
                if c > d:
                    MoV[c - 1] = temp_sum + math.floor(float(total_difference - lower_bound)/change[index][i]) + 1
                # if c < d, score(c) = score (d), the winner will change to d.
                else:
                    MoV[c - 1] = temp_sum + math.ceil(float(total_difference - lower_bound)/change[index][i])

                break
    return min(MoV)


def ScoringWinner(profile, scoringVector):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
        each position in a ranking from first to last.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    n = profile.numVoters
    m = profile.numCands
    if len(scoringVector) != m:
        print("ERROR: the length of the scoring vector is not correct!")
        exit()

    # Construct the score matrix--values
    prefcounts = profile.getPreferenceCounts()
    rankmaps = profile.getRankMaps()
    len_prefcounts = len(prefcounts)
    values = zeros([len_prefcounts, m], dtype=int)
    for i in range(len_prefcounts):
        values[i] = scoringVector[array(rankmaps[i].values()) - 1]

    return argmax(dot(array(prefcounts), values), axis=0) + 1
    
def MoVPlurality(profile):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest Borda score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
        each position in a ranking from first to last.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    scoringVector = []
    for i in range(m):
        scoringVector.append(0)
    scoringVector[0] = 1
    MoV = MoVScoring(profile, scoringVector)
    return MoV


def MoVBorda(profile):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest Borda score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
        each position in a ranking from first to last.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    scoringVector = array(range(m - 1, -1, -1))
    MoV = MoVScoring(profile, scoringVector)
    return MoV


def BordaWinner(profile):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    scoringVector = array(range(m - 1, -1, -1))
    winner = ScoringWinner(profile, scoringVector)
    return winner


def MoVVeto(profile):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest Veto score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
        each position in a ranking from first to last.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    scoringVector = ones(m, dtype=int)
    scoringVector[m - 1] = 0
    MoV = MoVScoring(profile, scoringVector)
    return MoV


def VetoWinner(profile):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    scoringVector = ones(m, dtype=int)
    scoringVector[m - 1] = 0
    winner = ScoringWinner(profile, scoringVector)
    return winner


def MoVkApproval(profile, k):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest Veto score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
        each position in a ranking from first to last.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    scoringVector = zeros(m, dtype=int)
    scoringVector[0:k] = 1
    MoV = MoVScoring(profile, scoringVector)
    return MoV


def kApprovalWinner(profile, k):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the largest score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    """
    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    scoringVector = zeros(m, dtype=int)
    scoringVector[0:k] = 1
    winner = ScoringWinner(profile, scoringVector)
    return winner

# def movSimplifiedBucklin(profile):
#     """
#     Returns an integer that is equal to the margin of victory of the election profile, that is,
#     the number of votes needed to be changed to change the winner.
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     """
#
#     # Currently, we expect the profile to contain strict complete ordering over candidates.
#     elecType = profile.getElecType()
#     if elecType != "soc":
#         print("ERROR: unsupported profile type")
#         exit()
#
#     # See if the election ends in a tie. If so, the mov is 0.
#     from . import mechanism
#     bucklin = mechanism.MechanismSimplifiedBucklin()
#     winners = bucklin.getWinners(profile)
#     if len(winners) > 1:
#         return 1
#
#     rankMaps = profile.getRankMaps()
#     preferenceCounts = profile.getPreferenceCounts()
#     winner = winners[0]
#
#     # Create a two-dimensional dictionary that associates each candidate with the number of times
#     # she appears at each rank.
#     rankCounts = dict()
#     for cand in profile.candMap.keys():
#
#         # Initialize the interior dictionary.
#         rankCount = dict()
#         for i in range(1, profile.numCands+1):
#             rankCount[i] = 0
#
#         # Fill the interior dictionary with the number of times the current candidate appears at
#         # each possible position.
#         for i in range(0, len(rankMaps)):
#             rank = rankMaps[i][cand]
#             rankCount[rank] += preferenceCounts[i]
#         rankCounts[cand] = rankCount
#
#     mov = float('inf')
#     for cand in profile.candMap.keys():
#         if cand == winner:
#             continue
#
#         # Integers to track the number of times the current candidate is in the top l positions,
#         # and the number of times the winning candidate is in the top l-1 positions.
#         candTopLCount = rankCounts[cand][1]
#         winnerTopLCount = 0
#         for l in range(2, int((profile.numCands+1)/2+1)):
#             candTopLCount += rankCounts[cand][l]
#             winnerTopLCount += rankCounts[winner][l-1]
#
#             # Calculate the minimum number of votes changed needed to make the current candidate be
#             # ranked in the top l positions in more than half the votes and make the winning
#             # candidate be ranked in the top l-1 votes in less than half the votes.
#             candVotesChanged = max(0, (profile.numVoters/2+1)-candTopLCount)
#             winnerVotesChanged = max(0, (profile.numVoters/2+1)-(profile.numVoters-winnerTopLCount))
#
#             # The margin of victory is the minimum number of votes needed for the above to occur
#             # given any candidate and any two positions l and l-1.
#             mov = min(mov, max(candVotesChanged, winnerVotesChanged))
#
#     return int(mov)

def MoVSimplifiedBucklin(profile):
    """
    Returns an integer that is equal to the margin of victory of the election profile, that is,
    the smallest number k such that changing k votes can change the winners.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    # Initialization
    n = profile.numVoters
    m = profile.numCands
    half = math.floor(float(n) / 2)
    prefcounts = profile.getPreferenceCounts()
    len_prefcounts = len(prefcounts)
    rankmaps = profile.getRankMaps()
    values = zeros([len_prefcounts, m], dtype=int)
    for i in range(len_prefcounts):
        values[i] = rankmaps[i].values()

    d = BucklinWinner(profile)
    alter = delete(range(1, m + 1), d - 1)
    # Initialize MoV
    MoV = n * ones(m, dtype=int)
    for c in alter:
        for ell in range(1, int(math.floor(float(m) / 2)) + 2):
            numcond1 = sum(dot(array(prefcounts), logical_and(values[:, c - 1] > ell, values[:, d - 1] <= ell - 1)))
            numcond2 = sum(dot(array(prefcounts), logical_and(values[:, c - 1] > ell, values[:, d - 1] > ell - 1)))
            numcond3 = sum(dot(array(prefcounts), logical_and(values[:, c - 1] <= ell, values[:, d - 1] <= ell - 1)))
            diff_c = half - sum(dot(array(prefcounts), (values[:, c - 1] <= ell)))
            diff_d = half - sum(dot(array(prefcounts), (values[:, d - 1] <= ell - 1)))

            if diff_c < 0:
                if diff_d < 0 and numcond1 + numcond3 > abs(diff_d):
                    MoV[c - 1] = min(MoV[c - 1], abs(diff_d))
                continue
            # -------diff_c >= 0------------
            if diff_d >= 0:
                if numcond1 + numcond2 > diff_c >= 0:
                    MoV[c - 1] = min(MoV[c - 1], diff_c + 1)
            else:
                if numcond1 > diff_c and numcond1 > abs(diff_d):
                    MoV[c - 1] = min(MoV[c - 1], max(diff_c + 1, abs(diff_d)))
                elif diff_c >= numcond1 > abs(diff_d):
                    if numcond1 + numcond2 > diff_c:
                        MoV[c - 1] = min(MoV[c - 1], diff_c + 1)
                elif abs(diff_d) >= numcond1 > diff_c:
                    if numcond1 + numcond3 > abs(diff_d):
                        MoV[c - 1] = min(MoV[c - 1], abs(diff_d))
                else:  # numcond1 <= diff_c and numcond1 <= abs(diff_d)
                    if numcond1 + numcond2 > diff_c and numcond1 + numcond3 > abs(diff_d):
                        MoV[c - 1] = min(MoV[c - 1], numcond1 + abs(diff_c) + 1 + abs(diff_d))

    return min(MoV)


def BucklinWinner(profile):
    """
    Returns an integer that represents the winning candidate given an election profile.
    The winner has the smallest Bucklin score.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    BucklinScores = zeros(m, dtype=int)
    for c in range(1, m + 1):
        BucklinScores[c - 1] = BucklinScore(profile, c)

    return argmin(BucklinScores) + 1


def BucklinScore(profile, c):
    """
    Returns an integer that is equal to the alternative c's Bucklin score.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    n = profile.numVoters
    m = profile.numCands
    prefcounts = profile.getPreferenceCounts()
    rankmaps = profile.getRankMaps()
    len_prefcounts = len(prefcounts)
    values = zeros([len_prefcounts, m], dtype=int)
    for i in range(len_prefcounts):
        values[i] = rankmaps[i].values()

    for ell in range(1, m + 1):
        num_c = sum(dot(array(prefcounts), (values[:, c - 1] <= ell)))
        if num_c > math.floor(float(n) / 2):
            return ell

def MoVPluRunOff(profile):
    """
    Returns an integer that is equal to the margin of victory of the election profile, that is,
    the smallest number k such that changing k votes can change the winners.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    # Initialization
    n = profile.numVoters
    m = profile.numCands
    half = math.floor(float(n) / 2)
    prefcounts = profile.getPreferenceCounts()
    len_prefcounts = len(prefcounts)
    rankmaps = profile.getRankMaps()
    values = zeros([len_prefcounts, m], dtype=int)
    for i in range(len_prefcounts):
        values[i] = rankmaps[i].values()

    # 1st round: find the top 2 candidates in plurality score
    plu = dot(array(prefcounts), values == 1)
    plu_1 = copy(plu)
    # Compute the 1st-place candidate in plurality score
    max_cand = argmax(plu_1) + 1

    plu_1[max_cand - 1] = 0
    # Compute the 2nd-place candidate in plurality score
    # Automatically using tie-breaking rule--numerically increasing order
    second_max_cand = argmax(plu_1) + 1

    plu_1[second_max_cand - 1] = 0
    # Compute the 3rd-place candidate (c2) in plurality score
    # Automatically using tie-breaking rule--numerically increasing order
    c2 = argmax(plu_1) + 1

    # 2nd round: find the candidate with maximum plurality score
    dict_top2 = {}
    dict_top2[max_cand] = values[:, max_cand - 1]
    dict_top2[second_max_cand] = values[:, second_max_cand - 1]
    rankmat = array(dict_top2.values()).transpose()

    plu_2 = {}
    for i in dict_top2.keys():
        plu_2[i] = 0
        for j in range(len(prefcounts)):
            plu_2[i] += prefcounts[j] * (dict_top2[i][j] == min(rankmat[j, :]))

    d = max(plu_2.items(), key=lambda x: x[1])[0]
    # the alternative who has the highest plurality score in C\{d}
    c1 = min(plu_2.items(), key=lambda x: x[1])[0]

    wmg = profile.getWmg(normalize=False)

    for k in range(1, n + 1):
        # Check 1
        if max(plu[d - 1] - k - plu[c1 - 1], 0) + max(plu[d - 1] - k - plu[c2 - 1], 0) <= k:
            return k
        # Check 2
        alter = range(1, m + 1)
        alter.remove(d)
        for c in alter:
            alter.remove(c)
            for ell in range(plu[c - 1] + k):

                summation = 0

                for e in alter:
                    t_e_ell = max(plu[e - 1] - ell, 0)
                    A_e = sum(dot(array(prefcounts), logical_and(values[:, e - 1] == 1, values[:, d - 1] < values[:, c - 1])))
                    summation += min(t_e_ell, A_e) - t_e_ell

                D_Q = wmg[c][d] + 2 * (k + summation)
                if D_Q >= 0:
                    return k


def PluRunOffWinner(profile):
    """
    Returns an integer that represents the winning candidate given an election profile.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    # Initialization
    n = profile.numVoters
    m = profile.numCands
    half = math.floor(float(n) / 2)
    prefcounts = profile.getPreferenceCounts()
    len_prefcounts = len(prefcounts)
    rankmaps = profile.getRankMaps()
    values = zeros([len_prefcounts, m], dtype=int)
    for i in range(len_prefcounts):
        values[i] = rankmaps[i].values()

    # 1st round: find the top 2 candidates in plurality score
    plu_1 = dot(array(prefcounts), values == 1)
    # Compute the 1st-place candidate in plurality score
    max_cand = argmax(plu_1) + 1

    plu_1[max_cand - 1] = 0
    # Compute the 2nd-place candidate in plurality score
    # Automatically using tie-breaking rule--numerically increasing order
    second_max_cand = argmax(plu_1) + 1

    # 2nd round: find the candidate with maximum plurality score
    dict_top2 = {}
    dict_top2[max_cand] = values[:, max_cand - 1]
    dict_top2[second_max_cand] = values[:, second_max_cand - 1]
    rankmat = array(dict_top2.values()).transpose()

    plu_2 = {}
    for i in dict_top2.keys():
        plu_2[i] = 0
        for j in range(len(prefcounts)):
            plu_2[i] += prefcounts[j] * (dict_top2[i][j] == min(rankmat[j, :]))

    return max(plu_2.items(), key=lambda x: x[1])[0]

def AppMoVMaximin(profile):
    """
    Returns an integer that is equal to the margin of victory of the election profile, that is,
    the smallest number k such that changing k votes can change the winners.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    # Initialization
    n = profile.numVoters
    m = profile.numCands

    # Compute the original winner d
    wmgMap = profile.getWmg()
    # Initialize each Copeland score as infinity.
    maximinscores = {}
    for cand in wmgMap.keys():
        maximinscores[cand] = float("inf")

    # For each pair of candidates, calculate the number of votes in which one beat the other.

    # For each pair of candidates, calculate the number of times each beats the other.
    for cand1, cand2 in itertools.combinations(wmgMap.keys(), 2):
        if cand2 in wmgMap[cand1].keys():
            maximinscores[cand1] = min(maximinscores[cand1], wmgMap[cand1][cand2])
            maximinscores[cand2] = min(maximinscores[cand2], wmgMap[cand2][cand1])
    d = max(maximinscores.items(), key=lambda x: x[1])[0]

    #Compute c* = argmax_c maximinscores(c)
    scores_without_d = maximinscores.copy()
    del scores_without_d[d]

    c_star = max(scores_without_d.items(), key=lambda x: x[1])[0]

    return (maximinscores[d] - maximinscores[c_star])/2


def MaximinWinner(profile):
    """
    Returns an integer that represents the winning candidate given an election profile.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    maximinscores = getMaximinScores(profile)
    winner = max(maximinscores.items(), key=lambda x: x[1])[0]
    return winner


def getMaximinScores(profile):
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

    wmgMap = profile.getWmg()
    # Initialize each Copeland score as infinity.
    maximinscores = {}
    for cand in wmgMap.keys():
        maximinscores[cand] = float("inf")

    # For each pair of candidates, calculate the number of votes in which one beat the other.

    # For each pair of candidates, calculate the number of times each beats the other.
    for cand1, cand2 in itertools.combinations(wmgMap.keys(), 2):
        if cand2 in wmgMap[cand1].keys():
            maximinscores[cand1] = min(maximinscores[cand1], wmgMap[cand1][cand2])
            maximinscores[cand2] = min(maximinscores[cand2], wmgMap[cand2][cand1])

    return maximinscores


def AppMoVCopeland(profile, alpha=0.5):
    """
    Returns an integer that is equal to the margin of victory of the election profile, that is,
    the smallest number k such that changing k votes can change the winners.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    # Initialization
    n = profile.numVoters
    m = profile.numCands

    # Compute the original winner d
    # Initialize each Copeland score as 0.0.
    copelandscores = {}
    for cand in profile.candMap.keys():
        copelandscores[cand] = 0.0

    # For each pair of candidates, calculate the number of votes in which one beat the other.
    wmgMap = profile.getWmg()
    for cand1, cand2 in itertools.combinations(wmgMap.keys(), 2):
        if cand2 in wmgMap[cand1].keys():
            if wmgMap[cand1][cand2] > 0:
                copelandscores[cand1] += 1.0
            elif wmgMap[cand1][cand2] < 0:
                copelandscores[cand2] += 1.0

            # If a pair of candidates is tied, we add alpha to their score for each vote.
            else:
                copelandscores[cand1] += alpha
                copelandscores[cand2] += alpha
    d = max(copelandscores.items(), key=lambda x: x[1])[0]

    #Compute c* = argmin_c RM(d,c)
    relative_margin = {}
    alter_without_d = delete(range(1, m + 1), d - 1)
    for c in alter_without_d:
        relative_margin[c] = RM(wmgMap, n, m, d, c, alpha)
    c_star = min(relative_margin.items(), key=lambda x: x[1])[0]

    return relative_margin[c_star]*(math.ceil(log(m)) + 1)


def RM(wmgMap, n, m, d, c, alpha=0.5):

    alter_without_d = delete(range(1, m + 1), d - 1)
    alter_without_c = delete(range(1, m + 1), c - 1)
    for t in range(n):
        # Compute s_-t_d and s_t_c
        s_neg_t_d = 0
        s_t_c = 0
        for e in alter_without_d:
            if wmgMap[e][d] < -2 * t:
                s_neg_t_d += 1.0
            elif wmgMap[e][d] == -2 * t:
                s_neg_t_d += alpha
        for e in alter_without_c:
            if wmgMap[e][c] < 2 * t:
                s_t_c += 1.0
            elif wmgMap[e][c] == 2 * t:
                s_t_c += alpha

        if s_neg_t_d <= s_t_c:
            return t


def CopelandWinner(profile, alpha=0.5):
    """
    Returns an integer that represents the winning candidate given an election profile.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    copelandscores = getCopelandScores(profile, alpha)
    winner = max(copelandscores.items(), key=lambda x: x[1])[0]
    return winner


def getCopelandScores(profile, alpha=0.5):
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
    copelandscores = {}
    for cand in profile.candMap.keys():
        copelandscores[cand] = 0.0

    # For each pair of candidates, calculate the number of votes in which one beat the other.
    wmgMap = profile.getWmg()
    for cand1, cand2 in itertools.combinations(wmgMap.keys(), 2):
        if cand2 in wmgMap[cand1].keys():
            if wmgMap[cand1][cand2] > 0:
                copelandscores[cand1] += 1.0
            elif wmgMap[cand1][cand2] < 0:
                copelandscores[cand2] += 1.0

            # If a pair of candidates is tied, we add alpha to their score for each vote.
            else:
                copelandscores[cand1] += alpha
                copelandscores[cand2] += alpha

    return copelandscores

