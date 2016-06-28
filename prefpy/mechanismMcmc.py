"""
Author: Kevin J. Hwang
"""
import io
import math
import itertools
import copy
import random
from profile import Profile
from preference import Preference
from mechanism import Mechanism

class MechanismMcmc(Mechanism):
           
    def getCandScoresMap(self, profile):
        """
        Returns a dictonary that associates the integer representation of each candidate with the 
        Bayesian utilities we approximate from our sampling of the profile.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        wmg = profile.getWmg(True)
        V = self.getInitialSample(wmg)

        utilities = dict()
        for cand in profile.candMap.keys():
            utilities[cand] = 0.0

        for i in range(0, self.burnIn):
            V = self.sampleGenerator.getNextSample(V)

        for i in range(0, self.n2):
            for j in range(0, self.n1):
                V = self.sampleGenerator.getNextSample(V)
            for cand in profile.candMap.keys():
                utilities[cand] += self.utilityFunction.getUtility([cand], V)

        for cand in profile.candMap.keys():
            utilities[cand] = utilities[cand]/self.n2

        return utilities

    def getWinnersBruteForce(self, profile):
        """
        Returns a list of all winning candidates when we use brute force to compute Bayesian
        utilities for an election profile. This function assumes that 
        getCandScoresMapBruteForce(profile) is implemented for the child MechanismMcmc class.
        
        :ivar Profile profile: A Profile object that represents an election profile.
        """

        candScoresMapBruteForce = self.getCandScoresMapBruteForce(profile) 

        # Check whether the winning candidate is the candidate that maximizes the score or 
        # minimizes it.
        if self.maximizeCandScore == True:
            bestScore = max(candScoresMapBruteForce.values())
        else:
            bestScore = min(candScoresMapBruteForce.values())
        
        # Create a list of all candidates with the winning score and return it.
        winners = []
        for cand in candScoresMapBruteForce.keys():
            if candScoresMapBruteForce[cand] == bestScore:
                winners.append(cand)
        return winners

    def getRankingBruteForce(self, profile):
        """
        Returns a list that orders all candidates from best to worst when we use brute force to 
        compute Bayesian utilities for an election profile. This function assumes that 
        getCandScoresMapBruteForce(profile) is implemented for the child Mechanism class. Note that
        the returned list gives no indication of ties between candidates. 
        
        :ivar Profile profile: A Profile object that represents an election profile.
        """

        # We generate a map that associates each score with the candidates that have that score.
        candScoresMapBruteForce = self.getCandScoresMapBruteForce(profile) 
        reverseCandScoresMap = dict()
        for key, value in candScoresMapBruteForce.items():
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

class MechanismMcmcMallows(MechanismMcmc):

    def __init__(self, phi, lossFunction, n1, n2, burnIn, sampleGenerator):
        self.maximizeCandScore = False
        self.phi = phi
        self.utilityFunction = lossFunction
        self.n1 = n1
        self.n2 = n2
        self.burnIn = burnIn
        self.sampleGenerator = sampleGenerator

    def kendallTau(self, orderVector, wmgMap):
        """
        Given a ranking for a single vote and a wmg for the entire election, calculate the kendall-tau
        distance. a.k.a the number of discordant pairs between the wmg for the vote and the wmg for the
        election. Currently, we expect the vote to be a strict complete ordering over the candidates.

        :ivar list<int> rankList: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last.
        :ivar dict<int,<dict,<int,int>>> wmgMap: A two-dimensional dictionary that associates integer
            representations of each pair of candidates, cand1 and cand2, with the number of times
            cand1 is ranked above cand2 minus the number of times cand2 is ranked above cand1. The
            dictionary represents a weighted majority graph constructed from an entire election.
        """

        discordantPairs = 0.0
        for i in itertools.combinations(orderVector, 2):
            discordantPairs = discordantPairs + max(0, wmgMap[i[1]][i[0]])
        return discordantPairs

    def getInitialSample(self, wmg):
        """
        Generate an initial sample for the Markov chain. This function will return a list 
        containing integer representations of each candidate in order of their rank in the current 
        vote, from first to last. The list will be a complete strict ordering over the candidates.
        Initially, we rank the candidates in random order.

        ivar: dict<int,<dict,<int,int>>> wmg: A two-dimensional dictionary that associates integer
            representations of each pair of candidates, cand1 and cand2, with the number of times
            cand1 is ranked above cand2 minus the number of times cand2 is ranked above cand1. The
            dictionary represents a weighted majority graph for an election.
        """
        
        V = copy.deepcopy(wmg.keys())
        random.shuffle(V)
        return V

    def getCandScoresMapBruteForce(self, profile):
        wmg = profile.getWmg(True)
        losses = dict()
        for cand in wmg.keys():
            losses[cand] = 0.0

        # Calculate the denominator.
        denom = 0.0
        for permutation in itertools.permutations(wmg.keys()):
            denom = denom + self.phi ** float(self.kendallTau(permutation, wmg))

        for permutation in itertools.permutations(wmg.keys()):
            prob = self.phi**float(self.kendallTau(permutation, wmg))/denom
            for cand in wmg.keys():
                losses[cand] += self.utilityFunction.getUtility([cand], permutation)* prob
        return losses


   