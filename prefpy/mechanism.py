"""
Author: Kevin J. Hwang
"""
import io
import math
import itertools
from profile import Profile
from preference import Preference

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
        Returns a list that orders all candidates from best to worst given an election profile.
        This function assumes that getCandScoresMap(profile) is implemented for the child Mechanism
        class. Note that the returned list gives no indication of ties between candidates. 
        
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
            for cand in reverseCandScoresMap[candScore]:
                ranking.append(cand)

        return ranking

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

        # Currently, we expect the profile to contain strict complete ordering over candidates.
        elecType = profile.getElecType()
        if elecType != "soc":
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

        # Currently, we expect the profile to contain strict complete ordering over candidates.
        elecType = profile.getElecType()
        if elecType != "soc":
            print("ERROR: unsupported election type")
            exit()

        # See if the election ends in a tie. If so, the mov is 0.
        winners = self.getWinners(profile)
        if len(winners) > 1:
            return 0

        rankMaps = profile.getRankMaps()
        scoringVector = self.getScoringVector(profile)
        preferenceCounts = profile.getPreferenceCounts()
        winner = winners[0]
        candScoresMap = self.getCandScoresMap(profile)
        mov = float('inf')

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
                votesNeeded += math.ceil(float(winnerScore-candScore)/float(ttlChange))
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

        scoringVector = []
        for i in range(0, profile.numCands-1):
            scoringVector.append(1)
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

        # Currently, we expect the profile to contain strict complete ordering over candidates.
        elecType = profile.getElecType()
        if elecType != "soc":
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

        # Currently, we expect the profile to contain strict complete ordering over candidates.
        elecType = profile.getElecType()
        if elecType != "soc":
            print("ERROR: unsupported profile type")
            exit()
        
        # See if the election ends in a tie. If so, the mov is 0.
        winners = self.getWinners(profile)
        if len(winners) > 1:
            return 0

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

class MechanismCopeland(Mechanism):
    """
    The Copeland mechanism.
    """

    def __init__(self, alpha):
        self.maximizeCandScore = True
        self.alpha = True

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
        for i in range(0, len(profile.preferences)):
            wmgMap = profile.preferences[i].wmgMap
            for cand1, cand2 in itertools.combinations(wmgMap.keys(), 2):
                if cand2 in wmgMap[cand1].keys():
                    if wmgMap[cand1][cand2] > 0:
                        copelandScores[cand1] += 1.0*preferenceCounts[i]
                    elif wmgMap[cand1][cand2] < 0:
                        copelandScores[cand2] += 1.0*preferenceCounts[i]
            
                    #If a pair of candidates is tied, we add alpha to their score for each vote.
                    else:
                        copelandScores[cand1] += alpha*preferenceCounts[i]
                        copelandScores[cand2] += alpha*preferenceCounts[i]

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