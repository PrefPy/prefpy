"""
Author: Kevin J. Hwang
"""
import io
import math
import itertools
from .preference import Preference

def movPosScoring(profile, scoringVector):
    """
    Returns an integer that is equal to the margin of victory of a profile, that is, the number of
    votes needed to be changed to change to winner when using the positional scoring rule.

    :ivar Profile profile: A Profile object that represents an election profile.
    :ivar list<int> scoringVector: A list of integers (or floats) that give the scores assigned to
        each position in a ranking from first to last.
    """

    # Currently, we expect the profile to contain complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc" and elecType != "toc":
        print("ERROR: unsupported election type")
        exit()

    # If the profile already results in a tie, return 0.
    from . import mechanism
    posScoring = mechanism.MechanismPosScoring(scoringVector)
    winners = posScoring.getWinners(profile)
    if len(winners) > 1:
        return 1

    rankMaps = profile.getRankMaps()
    preferenceCounts = profile.getPreferenceCounts()
    winner = winners[0]
    candScoresMap = posScoring.getCandScoresMap(profile)
    mov = profile.numVoters
    # For each candidate, calculate the difference in scores that changing a vote can do.
    for cand in profile.candMap.keys():
        scoreEffects = []
        if cand == winner: continue
        for i in range(0, len(rankMaps)):
            rankMap = rankMaps[i]
            incInCandScore = scoringVector[0]-scoringVector[rankMap[cand]-1]
            decInWinnerScore = scoringVector[rankMap[winner]-1]-scoringVector[-1]

            # Create a tuple that contains the max increase/decrease and the rankmap count.
            scoreEffect = (incInCandScore, decInWinnerScore, preferenceCounts[i])
            scoreEffects.append(scoreEffect)

        scoreEffects = sorted(scoreEffects, key=lambda scoreEffect: scoreEffect[0] + scoreEffect[1], reverse = True)

        # We simulate the effects of changing the votes starting with the votes that will have the
        # greatest impact.
        winnerScore = candScoresMap[winner]
        candScore = candScoresMap[cand]
        votesNeeded = 0

        for i in range(0, len(scoreEffects)):
            scoreEffect = scoreEffects[i] 
            ttlChange = scoreEffect[0] + scoreEffect[1]


            # Check if changing all instances of the current vote can change the winner.
            if (ttlChange*scoreEffect[2] >= winnerScore-candScore):
                votesNeeded += math.ceil(float(winnerScore-candScore)/float(ttlChange))+1
                break

            # Otherwise, update the election simulation with the effects of the current votes.
            else:
                votesNeeded += scoreEffect[2]
                winnerScore -= scoreEffect[1]*scoreEffect[2]
                candScore += scoreEffect[0]*scoreEffect[2]

            # If the number of votes needed to make the current candidate the winner is greater than
            # the lowest number of votes needed to make some candidate the winner, we can stop 
            # trying.
            if votesNeeded > mov:
                break
        
        mov = min(mov,votesNeeded)
        
    return int(mov)
    

def movPlurality(profile):
    """
    Returns an integer that is equal to the margin of victory, that is, the number of votes needed
    to be changed to change the winner when using the plurality rule.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    scoringVector = []
    scoringVector.append(1)
    for i in range(1, profile.numCands):
        scoringVector.append(0)
    #return movPosScoring(profile, scoringVector)
    return -1

def movVeto(profile):
    """
    Returns an integer that is equal to the margin of victory, that is, the number of votes needed
    to be changed to change the winner when using the veto rule.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    scoringVector = []
    for i in range(0, profile.numCands-1):
        scoringVector.append(1)
    scoringVector.append(0)
    return movPosScoring(profile, scoringVector)

def movBorda(profile):
    """
    Returns an integer that is equal to the margin of victory, that is, the number of votes needed
    to be changed to change the winner when using the veto rule.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    scoringVector = []
    score = profile.numCands-1
    for i in range(0, profile.numCands):
        scoringVector.append(score)
        score -= 1
    return movPosScoring(profile, scoringVector)
    

def movKApproval(profile, k):
    """
    Returns an integer that is equal to the margin of victory, that is, the number of votes needed
    to be changed to change the winner when using the k-approval rule.

    :ivar Profile profile: A Profile object that represents an election profile.
    :ivar int k: A value for k.
    """
    scoringVector = []
    for i in range(0, k):
        scoringVector.append(1)
    for i in range(k, profile.numCands):
        scoringVector.append(0)
    return movPosScoring(profile, scoringVector)

def movSimplifiedBucklin(profile):
    """
    Returns an integer that is equal to the margin of victory of the election profile, that is,
    the number of votes needed to be changed to change the winner.

    :ivar Profile profile: A Profile object that represents an election profile.
    """

    # Currently, we expect the profile to contain strict complete ordering over candidates.
    elecType = profile.getElecType()
    if elecType != "soc":
        print("ERROR: unsupported profile type")
        exit()
        
    # See if the election ends in a tie. If so, the mov is 0.
    bucklin = mechanism.MechanismSimplifiedBucklin()
    winners = bucklin.getWinners(profile)
    if len(winners) > 1:
        return 1

    rankMaps = profile.getRankMaps()
    preferenceCounts = profile.getPreferenceCounts()
    winner = winners[0]

    # Create a two-dimensional dictionary that associates each candidate with the number of times
    # she appears at each rank.
    rankCounts = dict()
    for cand in profile.candMap.keys():
        
        # Initialize the interior dictionary.
        rankCount = dict()
        for i in range(1, profile.numCands+1):
            rankCount[i] = 0
        
        # Fill the interior dictionary with the number of times the current candidate appears at
        # each possible position.
        for i in range(0, len(rankMaps)):
            rank = rankMaps[i][cand]
            rankCount[rank] += preferenceCounts[i]
        rankCounts[cand] = rankCount

    mov = float('inf')
    for cand in profile.candMap.keys():
        if cand == winner:
            continue

        # Integers to track the number of times the current candidate is in the top l positions,
        # and the number of times the winning candidate is in the top l-1 positions.
        candTopLCount = rankCounts[cand][1]
        winnerTopLCount = 0
        for l in range(2, int((profile.numCands+1)/2+1)):
            candTopLCount += rankCounts[cand][l]
            winnerTopLCount += rankCounts[winner][l-1]

            # Calculate the minimum number of votes changed needed to make the current candidate be
            # ranked in the top l positions in more than half the votes and make the winning 
            # candidate be ranked in the top l-1 votes in less than half the votes.
            candVotesChanged = max(0, (profile.numVoters/2+1)-candTopLCount)
            winnerVotesChanged = max(0, (profile.numVoters/2+1)-(profile.numVoters-winnerTopLCount))

            # The margin of victory is the minimum number of votes needed for the above to occur
            # given any candidate and any two positions l and l-1.
            mov = min(mov, max(candVotesChanged, winnerVotesChanged))
            
    return int(mov)



