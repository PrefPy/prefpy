"""
Authors: Kevin J. Hwang
        Jun Wang
"""
import prefpy_io
import math
import itertools
import copy
from numpy import *
from profile import Profile
from mechanism import *
from preference import Preference

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


# def MoVScoring_1(profile, scoringVector):
#     """
#     Returns an integer that represents the winning candidate given an election profile.
#     The winner has the largest score.
#     Tie-breaking rule: numerically increasing order
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     :ivar list<int> scoringVector: A list of integers (or floats) that give the scores assigned to
#         each position in a ranking from first to last.
#     """
#     # Currently, we expect the profile to contain complete ordering over candidates.
#     elecType = profile.getElecType()
#     if elecType != "soc" and elecType != "csv" and elecType != "toc":
#         print("ERROR: unsupported profile type")
#         exit()
#
#     n = profile.numVoters
#     m = profile.numCands
#     if len(scoringVector) != m:
#         print("ERROR: the length of the scoring vector is not correct!")
#         exit()
#
#     # Construct the score matrix--values
#     prefcounts = array(profile.getPreferenceCounts())
#     rankmaps = profile.getRankMaps()
#     len_prefcounts = len(prefcounts)
#     values = zeros([len_prefcounts, m], dtype=int)
#
#     if min(list(rankmaps[0].keys())) == 0:
#         delta = 0
#     else:
#         delta = 1
#
#     for i in range(len_prefcounts):
#         for j in range(delta, m + delta):
#             values[i][j - delta] = scoringVector[rankmaps[i][j] - 1]
#
#     # Compute the scores of all the candidates
#     score = dot(array(prefcounts), values)
#     # Compute the winner of the original profile
#     d = argmax(score, axis=0) + delta
#     # print("d=",d)
#     alter = delete(range(delta, m + delta), d - delta)
#     # Initialize
#     MoV = n * ones(m, dtype=int)
#
#     for c in alter:
#         # The difference vector of d and c
#         difference = values[:, c - delta] - values[:, d - delta]
#         index = argsort(difference, axis=0, kind='mergesort')
#         # The vector that each element is the gain in the difference
#         # between d and c if the pattern of the vote changed to [c > others > d]
#         change = scoringVector[0] - difference
#
#         # The total_difference between score(d) and score(c)
#         total_difference = score[d - delta] - score[c - delta]
#         for i in range(len_prefcounts):
#             # The number of votes of the first i kinds of patterns
#             temp_sum = sum(prefcounts[index][0:i])
#
#             # The aggregate gain (of the first i kinds of patterns)
#             # in the difference between d and c if changed to [c > others > d]
#             lower_bound = dot(prefcounts[index][0:i], change[index][0:i])
#
#             # The aggregate gain (of the first i+1 kinds of patterns)
#             # in the difference between d and c if changed to [c > others > d]
#             upper_bound = dot(prefcounts[index][0:i + 1], change[index][0:i + 1])
#             if lower_bound < total_difference <= upper_bound:
#                 # tie-breaking rule: numerically increasing order
#                 # if c > d, score(c) = score (d), the winner is still d,
#                 # so only when score(c) is strictly less than score(d), will the winner change to c.
#                 if c > d:
#                     MoV[c - delta] = temp_sum + math.floor(float(total_difference - lower_bound)/change[index][i]) + 1
#                 # if c < d, score(c) = score (d), the winner will change to c.
#                 else:
#                     MoV[c - delta] = temp_sum + math.ceil(float(total_difference - lower_bound)/change[index][i])
#
#                 break
#
#     return min(MoV)


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
    if elecType != "soc" and elecType != "csv" and elecType != "toc":
        print("ERROR: unsupported profile type")
        exit()

    winners = MechanismPosScoring(scoringVector).getWinners(profile)
    if len(winners) > 1:
        return 1
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

    if min(list(rankmaps[0].keys())) == 0:
        delta = 0
    else:
        delta = 1

    for i in range(len_prefcounts):
        for j in range(delta, m + delta):
            values[i][j - delta] = scoringVector[rankmaps[i][j] - 1]

    # Compute the scores of all the candidates
    score = dot(array(prefcounts), values)
    # Compute the winner of the original profile

    d = argmax(score, axis=0) + delta
    # print("d=",d)
    alter = delete(range(delta, m + delta), d - delta)
    # Initialize
    MoV = n * ones(m, dtype=int)
    # for c in [3]:
    for c in alter:
        # The difference vector of d and c
        difference = values[:, c - delta] - values[:, d - delta]
        # print("dif=", difference)
        index = argsort(difference, axis=0, kind='mergesort')
        # The vector that each element is the gain in the difference
        # between d and c if the pattern of the vote changed to [c > others > d]
        change = scoringVector[0] - difference

        # The total_difference between score(d) and score(c)
        total_difference = score[d - delta] - score[c - delta]
        # print("total-dif=", total_difference)
        for i in range(len_prefcounts):
            # The number of votes of the first i kinds of patterns
            temp_sum = sum(prefcounts[index][0:i])
            # print("temp_sum=", temp_sum)

            # The aggregate gain (of the first i kinds of patterns)
            # in the difference between d and c if changed to [c > others > d]
            lower_bound = dot(prefcounts[index][0:i], change[index][0:i])
            # print("lower_bound=", lower_bound)

            # The aggregate gain (of the first i+1 kinds of patterns)
            # in the difference between d and c if changed to [c > others > d]
            upper_bound = dot(prefcounts[index][0:i + 1], change[index][0:i + 1])
            # print("upper_bound=", upper_bound)
            # if lower_bound < total_difference <= upper_bound:
            if lower_bound <= total_difference < upper_bound:
                MoV[c - delta] = temp_sum + math.floor(float(total_difference - lower_bound)/change[index][i]) + 1
                break
    # print("MoV=", MoV)
    return min(MoV)

# def MoVPlurality(profile):
#     """
#     Returns an integer that represents the winning candidate given an election profile.
#     The winner has the largest Borda score.
#     Tie-breaking rule: numerically increasing order
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
#         each position in a ranking from first to last.
#     """
#     # Currently, we expect the profile to contain complete ordering over candidates.
#     elecType = profile.getElecType()
#     if elecType != "soc" and elecType != "toc":
#         print("ERROR: unsupported profile type")
#         exit()
#
#     winners = MechanismPlurality().getWinners(profile)
#     if len(winners) > 1:
#         return 1
#     m = profile.numCands
#     scoringVector = zeros(m, dtype=int)
#     scoringVector[0] = 1
#     # print(scoringVector)
#     MoV = MoVScoring(profile, scoringVector)
#     return MoV
#
#
# def MoVBorda(profile):
#     """
#     Returns an integer that represents the winning candidate given an election profile.
#     The winner has the largest Borda score.
#     Tie-breaking rule: numerically increasing order
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
#         each position in a ranking from first to last.
#     """
#     # Currently, we expect the profile to contain complete ordering over candidates.
#     elecType = profile.getElecType()
#     if elecType != "soc" and elecType != "toc":
#         print("ERROR: unsupported profile type")
#         exit()
#
#     winners = MechanismBorda().getWinners(profile)
#     if len(winners) > 1:
#         return 1
#     m = profile.numCands
#     scoringVector = array(range(m - 1, -1, -1))
#     MoV = MoVScoring(profile, scoringVector)
#     return MoV
#
#
# def MoVVeto(profile):
#     """
#     Returns an integer that represents the winning candidate given an election profile.
#     The winner has the largest Veto score.
#     Tie-breaking rule: numerically increasing order
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
#         each position in a ranking from first to last.
#     """
#     # Currently, we expect the profile to contain complete ordering over candidates.
#     elecType = profile.getElecType()
#     if elecType != "soc" and elecType != "toc":
#         print("ERROR: unsupported profile type")
#         exit()
#
#     winners = MechanismVeto().getWinners(profile)
#     if len(winners) > 1:
#         return 1
#     m = profile.numCands
#     scoringVector = ones(m, dtype=int)
#     scoringVector[m - 1] = 0
#     MoV = MoVScoring(profile, scoringVector)
#     return MoV
#
#
# def MoVkApproval(profile, k):
#     """
#     Returns an integer that represents the winning candidate given an election profile.
#     The winner has the largest Veto score.
#     Tie-breaking rule: numerically increasing order
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     :ivar array<int> scoringVector: A list of integers (or floats) that give the scores assigned to
#         each position in a ranking from first to last.
#     """
#     # Currently, we expect the profile to contain complete ordering over candidates.
#     elecType = profile.getElecType()
#     if elecType != "soc" and elecType != "toc":
#         print("ERROR: unsupported profile type")
#         exit()
#
#     winners = MechanismKApproval(k).getWinners(profile)
#     if len(winners) > 1:
#         return 1
#     m = profile.numCands
#     scoringVector = zeros(m, dtype=int)
#     scoringVector[0:k] = 1
#     MoV = MoVScoring(profile, scoringVector)
#     return MoV


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
    if elecType != "soc" and elecType != "csv" and elecType != "toc":
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
    if min(list(rankmaps[0].keys())) == 0:
        delta = 0
    else:
        delta = 1
    for i in range(len_prefcounts):
        for j in range(delta, m + delta):
            values[i][j - delta] = rankmaps[i][j]

    winners = MechanismSimplifiedBucklin().getWinners(profile)  # the winner list
    d = winners[0]  # the winner under the numerically tie-breaking rule
    alter = delete(range(delta, m + delta), d - delta)
    # Initialize MoV
    MoV = n * ones(m, dtype=int)
    for c in alter:
        for ell in range(1, int(math.floor(float(m) / 2)) + 2):
            numcond1 = sum(dot(array(prefcounts), logical_and(values[:, c - delta] > ell, values[:, d - delta] <= ell - 1)))
            numcond2 = sum(dot(array(prefcounts), logical_and(values[:, c - delta] > ell, values[:, d - delta] > ell - 1)))
            numcond3 = sum(dot(array(prefcounts), logical_and(values[:, c - delta] <= ell, values[:, d - delta] <= ell - 1)))
            diff_c = half - sum(dot(array(prefcounts), (values[:, c - delta] <= ell)))
            diff_d = half - sum(dot(array(prefcounts), (values[:, d - delta] <= ell - 1)))
            if diff_c < 0:
                if diff_d < 0 and numcond1 + numcond3 > abs(diff_d):
                    MoV[c - delta] = min(MoV[c - delta], abs(diff_d))
                continue
            # -------diff_c >= 0------------
            if diff_d >= 0:
                if numcond1 + numcond2 > diff_c >= 0:
                    MoV[c - delta] = min(MoV[c - delta], diff_c + 1)
            else:
                if numcond1 > diff_c and numcond1 > abs(diff_d):
                    MoV[c - delta] = min(MoV[c - delta], max(diff_c + 1, abs(diff_d)))
                elif diff_c >= numcond1 > abs(diff_d):
                    if numcond1 + numcond2 > diff_c:
                        MoV[c - delta] = min(MoV[c - delta], diff_c + 1)
                elif abs(diff_d) >= numcond1 > diff_c:
                    if numcond1 + numcond3 > abs(diff_d):
                        MoV[c - delta] = min(MoV[c - delta], abs(diff_d))
                else:  # numcond1 <= diff_c and numcond1 <= abs(diff_d)
                    if numcond1 + numcond2 > diff_c and numcond1 + numcond3 > abs(diff_d):
                        MoV[c - delta] = min(MoV[c - delta], numcond1 + abs(diff_c) + 1 + abs(diff_d))

    return min(MoV)


# def BucklinWinner(profile, values, delta=1):
#     """
#     Returns an integer that represents the winning candidate given an election profile.
#     The winner has the smallest Bucklin score.
#     Tie-breaking rule: numerically increasing order
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     """
#
#     m = profile.numCands
#     BucklinScores = zeros(m, dtype=int)
#     for c in range(delta, m + delta):
#         BucklinScores[c - delta] = BucklinScore(profile, c, values, delta)
#
#     return argmin(BucklinScores) + delta
#
#
# def BucklinScore(profile, c, values, delta=1):
#     """
#     Returns an integer that is equal to the alternative c's Bucklin score.
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     """
#
#     n = profile.numVoters
#     m = profile.numCands
#     prefcounts = profile.getPreferenceCounts()
#
#     for ell in range(1, m + 1):
#         num_c = sum(dot(array(prefcounts), (values[:, c - delta] <= ell)))
#         if num_c > math.floor(float(n) / 2):
#             return ell

# def MoVPluRunOff_1(profile):
#     """
#     Returns an integer that is equal to the margin of victory of the election profile, that is,
#     the smallest number k such that changing k votes can change the winners.
#
#     :ivar Profile profile: A Profile object that represents an election profile.
#     """
#
#     # Currently, we expect the profile to contain complete ordering over candidates.
#     elecType = profile.getElecType()
#     if elecType != "soc" and elecType != "toc" and elecType != "csv":
#         print("ERROR: unsupported profile type")
#         exit()
#
#     # Initialization
#     n = profile.numVoters
#     m = profile.numCands
#     prefcounts = profile.getPreferenceCounts()
#     len_prefcounts = len(prefcounts)
#     rankmaps = profile.getRankMaps()
#     values = zeros([len_prefcounts, m], dtype=int)
#     if min(list(rankmaps[0].keys())) == 0:
#         delta = 0
#     else:
#         delta = 1
#     for i in range(len_prefcounts):
#         for j in range(delta, m + delta):
#             values[i][j - delta] = rankmaps[i][j]
#
#     # print("values=", values)
#     # 1st round: find the top 2 candidates in plurality score
#     plu = dot(array(prefcounts), values == 1)
#     # print("plu=", plu)
#     plu_1 = copy.deepcopy(plu)
#     # Compute the 1st-place candidate in plurality score
#     max_cand = argmax(plu_1) + delta
#
#     plu_1[max_cand - delta] = 0
#     # Compute the 2nd-place candidate in plurality score
#     # Automatically using tie-breaking rule--numerically increasing order
#     second_max_cand = argmax(plu_1) + delta
#
#     plu_1[second_max_cand - delta] = 0
#     # Compute the 3rd-place candidate (c2) in plurality score
#     # Automatically using tie-breaking rule--numerically increasing order
#     c2 = argmax(plu_1) + delta
#
#     # 2nd round: find the candidate with maximum plurality score
#     dict_top2 = {}
#     rankmat = []
#     dict_top2[max_cand] = values[:, max_cand - delta]
#     rankmat.append(values[:, max_cand - delta])
#     dict_top2[second_max_cand] = values[:, second_max_cand - delta]
#     rankmat.append(values[:, second_max_cand - delta])
#     rankmat = array(rankmat).transpose()
#     # print("rankmat=\n", rankmat)
#
#     plu_2 = {}
#     for i in dict_top2.keys():
#         plu_2[i] = 0
#         for j in range(len(prefcounts)):
#             plu_2[i] += prefcounts[j] * (dict_top2[i][j] == min(rankmat[j, :]))
#
#     d = max(plu_2.items(), key=lambda x: x[1])[0]
#     # the alternative who has the highest plurality score in C\{d}
#     c1 = min(plu_2.items(), key=lambda x: x[1])[0]
#
#     wmg = profile.getWmg(normalize=False)
#     # print(wmg)
#
#     for k in range(1, n + 1):
#         # Check 1
#         if max(plu[d - delta] - k - plu[c1 - delta], 0) + max(plu[d - delta] - k - plu[c2 - delta], 0) <= k:
#             # print("k=", k)
#             return k
#         # Check 2
#         alter = list(range(delta, m + delta))
#         alter.remove(d)
#         for c in alter:
#             for ell in range(plu[c - delta] + k):
#                 summation = 0
#                 for e in set(alter).difference(set([c])):
#                     t_e_ell = max(plu[e - delta] - ell, 0)
#                     A_e = sum(dot(array(prefcounts), logical_and(values[:, e - delta] == 1, values[:, d - delta] < values[:, c - delta])))
#                     summation += min(t_e_ell, A_e) - t_e_ell
#
#                 D_Q = wmg[c][d] + 2 * (k + summation)
#                 if D_Q >= 0:
#                     # print("D_Q=", D_Q, "c=", c, "d=", d, "l=", ell)
#                     return k


def MoVPluRunOff(profile):
    """
    Returns an integer that is equal to the margin of victory of the election profile, that is,
    the smallest number k such that changing k votes can change the winners.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc" and elecType != "csv":
        print("ERROR: unsupported profile type")
        exit()

    # Initialization
    prefcounts = profile.getPreferenceCounts()
    len_prefcounts = len(prefcounts)
    rankmaps = profile.getRankMaps()
    # print(rankmaps)
    ranking = MechanismPlurality().getRanking(profile)
    # print("ranking=", ranking)

    # 1st round: find the top 2 candidates in plurality scores
    # Compute the 1st-place candidate in plurality scores
    max_cand = ranking[0][0][0]

    # Compute the 2nd-place candidate in plurality scores
    # Automatically using tie-breaking rule--numerically increasing order
    if len(ranking[0][0]) > 1:
        second_max_cand = ranking[0][0][1]
        if len(ranking[0][0]) > 2:
            third_max_cand = ranking[0][0][2]
        else:
            third_max_cand = ranking[0][1][0]
    else:
        second_max_cand = ranking[0][1][0]
        if len(ranking[0][1]) > 1:
            third_max_cand = ranking[0][1][1]
        else:
            third_max_cand = ranking[0][2][0]

    top_2 = [max_cand, second_max_cand]
    # 2nd round: find the candidate with maximum plurality score
    dict_top2 = {max_cand: 0, second_max_cand: 0}
    for i in range(len_prefcounts):
        vote_top2 = {key: value for key, value in rankmaps[i].items() if key in top_2}
        # print(vote_top2)
        top_position = min(vote_top2.values())
        keys = [x for x in vote_top2.keys() if vote_top2[x] == top_position]
        for key in keys:
            dict_top2[key] += prefcounts[i]

    # the original winner-- d
    # print("dict_top2=", dict_top2)
    d = max(dict_top2.items(), key=lambda x: x[1])[0]
    c_1 = top_2[0] if top_2[1] == d else top_2[1]
    # the candidate with third highest plurality score
    c_2 = third_max_cand
    # print("d=", d, c_1, c_2)

    Type1_1 = Type1_2 = 0
    plu_d = plu_c_1 = plu_c_2 = 0

    # ------------count the votes of CASE I & II---------------
    for i in range(len_prefcounts):
        if rankmaps[i][d] < rankmaps[i][c_1]:
            Type1_1 += prefcounts[i]
        elif rankmaps[i][d] > rankmaps[i][c_1]:
            Type1_2 += prefcounts[i]

        if rankmaps[i][d] == 1:
            plu_d += prefcounts[i]
        elif rankmaps[i][c_1] == 1:
            plu_c_1 += prefcounts[i]
        elif rankmaps[i][c_2] == 1:
            plu_c_2 += prefcounts[i]
    # print("plu=", plu_d, plu_c_1, plu_c_2)
    # -------------------CASE I------------------------------
    MoV_I = math.floor((Type1_1 - Type1_2)/2) + 1

    # -------------------CASE II-------------------------------
    if math.floor((plu_d + plu_c_2)/2) + 1 <= plu_c_1:
        MoV_II = math.floor((plu_d - plu_c_2)/2) + 1
    else:
        MoV_II = plu_d - math.floor((plu_d + plu_c_1 + plu_c_2)/3) + 1
        # MoV_II = math.floor((plu_d * 2 - plu_c_1 - plu_c_2) / 3) + 1  # old version

    # -------------------CASE III-----------------------------
    MoV_d = dict()
    remaining = sorted(rankmaps[0].keys())
    remaining.remove(d)
    remaining.remove(c_1)

    for e in remaining:
        # ------------count the votes of CASE III---------------
        T1 = T2 = T3 = T4 = T5 = T6 = T7 = T8 = 0
        for i in range(len_prefcounts):
            if rankmaps[i][d] == 1:
                if rankmaps[i][c_1] < rankmaps[i][e]:
                    T1 += prefcounts[i]
                elif rankmaps[i][e] < rankmaps[i][c_1]:
                    T2 += prefcounts[i]
            elif rankmaps[i][c_1] == 1:
                if rankmaps[i][d] < rankmaps[i][e]:
                    T3 += prefcounts[i]
                elif rankmaps[i][e] < rankmaps[i][d]:
                    T4 += prefcounts[i]
            elif rankmaps[i][e] == 1:
                if rankmaps[i][d] < rankmaps[i][c_1]:
                    T5 += prefcounts[i]
                elif rankmaps[i][c_1] < rankmaps[i][d]:
                    T6 += prefcounts[i]
            else:
                if rankmaps[i][d] < rankmaps[i][e]:
                    T7 += prefcounts[i]
                elif rankmaps[i][e] < rankmaps[i][d]:
                    T8 += prefcounts[i]

        if math.floor((T3 + T4 + T5 + T6)/2) + 1 <= T1 + T2:
            CHANGE1 = math.floor((T3 + T4 - T5 - T6)/2) + 1
        else:
            CHANGE1 = T3 + T4 - T1 -T2 + 1

        x = min(T3, CHANGE1)
        if T1 + T2 + T3 + T7 - x < T4 + T5 + T6 + T8 + x:
            MoV_d[e] = CHANGE1
        else:
            CHANGE2 = math.floor((T1 + T2 + T3 + T7 - T4 - T5 - T6 - T8)/2) - x + 1
            MoV_d[e] = CHANGE1 + CHANGE2

    MoV_III = min(MoV_d.items(), key=lambda x: x[1])[1]
    # ------------------------Overall MoV---------------------------------
    # print(MoV_d)
    # print(MoV_I, MoV_II, MoV_III)
    MoV = min(MoV_I, MoV_II, MoV_III)

    return MoV


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


def getCopelandScores(profile, alpha=0.5, normalize=False):
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

    if normalize:
        m = profile.numCands
        for cand in profile.candMap.keys():
            copelandscores[cand] /= (m - 1)

    return copelandscores


def MoV_SNTV(profile, K):
    """
    Returns an integer that represents the winning candidate given an election profile.
    Tie-breaking rule: numerically increasing order

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc" and elecType != "csv":
        print("ERROR: unsupported profile type")
        exit()

    m = profile.numCands
    candScoresMap = MechanismPlurality().getCandScoresMap(profile)
    if K >= m:
        return float("inf")
    # print(candScoresMap)
    sorted_items = sorted(candScoresMap.items(), key=lambda x: x[1], reverse=True)
    sorted_dict = {key: value for key, value in sorted_items}
    sorted_cand = list(sorted_dict.keys())
    MoV = math.floor((sorted_dict[sorted_cand[K - 1]] - sorted_dict[sorted_cand[K]]) / 2) + 1
    return MoV
