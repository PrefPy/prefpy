"""
Author: Kevin J. Hwang
"""
import io
import math
import itertools
import copy
import random
import json
from profile import Profile
from preference import Preference
from mechanism import Mechanism

SAMPLESFILEMETADATALINECOUNT = 3

class MechanismMcmc(Mechanism):

    def getWinners(self, profile, sampleFileName = None):
        """
        Returns a list of all winning candidates when we use MCMC approximation to compute Bayesian
        utilities for an election profile.
        
        :ivar Profile profile: A Profile object that represents an election profile.
        :ivar str sampleFileName: An optional argument for the name of the input file containing 
            sample data. If a file name is given, this method will use the samples in the file 
            instead of generating samples itself.
        """

        if sampleFileName != None:
            candScores = self.getCandScoresMapFromSamplesFile(profile, sampleFileName)
        else:
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

    def getRanking(self, profile, sampleFileName = None):
        """
        Returns a list of lists that orders all candidates in tiers from best to worst when we use 
        MCMC approximation to compute Bayesian utilities for an election profile.

        :ivar Profile profile: A Profile object that represents an election profile.
        :ivar str sampleFileName: An optional argument for the name of the input file containing 
            sample data. If a file name is given, this method will use the samples in the file 
            instead of generating samples itself.
        """

        if sampleFileName != None:
            candScoresMap = self.getCandScoresMapFromSamplesFile(profile, sampleFileName)
        else:
            candScoresMap = self.getCandScoresMap(profile)

        # We generate a map that associates each score with the candidates that have that acore.
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

    #-------------------Functions related to printing samples to a file and using them for mcmc----
    #-------------------approximation.-------------------------------------------------------------

    def getCandScoresMapFromSamplesFile(self, profile, sampleFileName):
        """
        Returns a dictonary that associates the integer representation of each candidate with the 
        Bayesian utilities we approximate from the samples we generated into a file.

        :ivar Profile profile: A Profile object that represents an election profile.
        :ivar str sampleFileName: The name of the input file containing the sample data.
        """

        wmg = profile.getWmg(True)

        # Initialize our list of expected utilities.
        utilities = dict()
        for cand in wmg.keys():
            utilities[cand] = 0.0

        # Open the file and skip the lines of meta data in the file and skip samples for burn-in.
        sampleFile = open(sampleFileName)
        for i in range(0, SAMPLESFILEMETADATALINECOUNT):
            sampleFile.readline()
        for i in range(0, self.burnIn):
            sampleFile.readline()

        # We update our utilities as we read the file.
        numSamples = 0
        for i in range(0, self.n2*self.n1):
            line = sampleFile.readline()
            if i % self.n1 != 0: continue
            sample = json.loads(line)
            for cand in wmg.keys():
                utilities[cand] += self.utilityFunction.getUtility([cand], sample)
            numSamples += 1
        sampleFile.close()
        for key in utilities.keys():
            utilities[key] = utilities[key]/numSamples

        return utilities

    def printMcmcSamplesToFile(self, profile, numSamples, outFileName):
        """
        Generate samples to a file.

        :ivar Profile profile: A Profile object that represents an election profile.
        :ivar int numSamples: The number of samples to be generated.
        :ivar str outFileName: The name of the file to be output.
        """

        wmg = profile.getWmg(True)
        V = self.getInitialSample(wmg)

        # Print the number of candidates, phi, and the number of samples.
        outFile = open(outFileName, 'w')
        outFile.write("m," + str(profile.numCands) + '\n')
        outFile.write("phi," + str(self.phi) + '\n')
        outFile.write("numSamples," + str(numSamples))
        
        for i in range(0, numSamples):
            V = self.sampleGenerator.getNextSample(V)
            outFile.write("\n" + json.dumps(V))
        outFile.close()

    #----------------------------------------------------------------------------------------------

class MechanismMcmcMallows(MechanismMcmc):
    """
    Implementation of the MCMC mechanism using the Mallows model.
    """

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
        """
        Returns a dictonary that associates the integer representation of each candidate with the 
        bayesian losses that we calculate using brute force.

        :ivar Profile profile: A Profile object that represents an election profile.
        """
        
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

class MechanismMcmcCondorcet(MechanismMcmc):
    """
    Implementation of the MCMC mechanism using the Condorcet model. This was mostly adapted from 
    code by Lirong Xia.
    """

    def __init__(self, phi, lossFunction, n1, n2, burnIn, sampleGenerator):
        self.maximizeCandScore = True
        self.phi = phi
        self.utilityFunction = lossFunction
        self.n1 = n1
        self.n2 = n2
        self.burnIn = burnIn
        self.sampleGenerator = sampleGenerator

    def createBinaryRelation(self, m):
        """
        Initialize a two-dimensional array of size m by m.
    
        :ivar int m: A value for m.
        """

        binaryRelation = []
        for i in range(m):
            binaryRelation.append(range(m))
            binaryRelation[i][i] = 0
        return binaryRelation

    def getInitialSample(self, wmg):
        """
        Generate an initial sample for the Markov chain. This function will return a 
        two-dimensional array of integers, such that for each pair of candidates, cand1 and cand2,
        the array contains 1 if more votes rank cand1 above cand2 and 0 otherwise.

        ivar: dict<int,<dict,<int,int>>> wmg: A two-dimensional dictionary that associates integer
            representations of each pair of candidates, cand1 and cand2, with the number of times
            cand1 is ranked above cand2 minus the number of times cand2 is ranked above cand1. The
            dictionary represents a weighted majority graph for an election.
        """

        cands = range(len(wmg))
        allPairs = itertools.combinations(cands, 2)
        V = self.createBinaryRelation(len(cands))
        for pair in allPairs:
            if wmg[pair[0]+1][pair[1]+1] > 0:
                V[pair[0]][pair[1]] = 1
                V[pair[1]][pair[0]] = 0
            else:
                V[pair[0]][pair[1]] = 0
                V[pair[1]][pair[0]] = 1
        return V

    def getCandScoresMapBruteForce(self, profile):
        """
        Returns a dictonary that associates the integer representation of each candidate with the 
        bayesian losses that we calculate using brute force.

        :ivar Profile profile: A Profile object that represents an election profile.
        """

        wmg = profile.getWmg(True)
        m = len(wmg.keys())
        cands = range(m)
        V = self.createBinaryRelation(m)
        gains = dict()
        for cand in wmg.keys():
            gains[cand] = 0
        graphs = itertools.product(range(2), repeat=m*(m-1)/2)
        for comb in graphs:
            prob = 1
            i = 0
            for a, b in itertools.combinations(cands,2):
                V[a][b] = comb[i]
                V[b][a] = 1-comb[i]
                if comb[i] > 0:
                    prob *= 1/(1+self.phi ** float(wmg[a+1][b+1]))
                else:
                    prob *= 1/(1+self.phi ** float(wmg[b+1][a+1]))
                i += 1
                if i >= m*(m-1)/2:
                    break
            for cand in wmg.keys():
                gains[cand] += self.utilityFunction.getUtility([cand], V)*prob
        return gains


