import numpy
import copy
import random
import itertools

class MechanismMcmcSampleGenerator():

    def __init__(self, wmg, phi):
        self.wmg = wmg
        self.phi = phi

    def setWmg(self, wmg):
        """
        Function to set the wmg. Some child classes may do this in a specific way.
        ivar: dict<int,<dict,<int,int>>> wmg: A two-dimensional dictionary that associates integer
            representations of each pair of candidates, cand1 and cand2, with the number of times
            cand1 is ranked above cand2 minus the number of times cand2 is ranked above cand1. The
            dictionary represents a weighted majority graph for an election.
        """

        self.wmg = wmg

    def setPhi(self, phi):
        """
        Function to set phi. Some child classes may do this in a specific way.

        :ivar float phi: A value for phi such that 0 <= phi <= 1.        
        """

        self.phi = phi

class MechanismMcmcSampleGeneratorMallows(MechanismMcmcSampleGenerator):
    
    def calcAcceptanceRatio(self, V, W):
        """
        Given a order vector V and a proposed order vector W, calculate the acceptance ratio for 
        changing to W when using MCMC.

        ivar: dict<int,<dict,<int,int>>> wmg: A two-dimensional dictionary that associates integer
            representations of each pair of candidates, cand1 and cand2, with the number of times
            cand1 is ranked above cand2 minus the number of times cand2 is ranked above cand1. The
            dictionary represents a weighted majority graph for an election.
        :ivar float phi: A value for phi such that 0 <= phi <= 1.   
        :ivar list<int> V: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last. This is the current sample.
        :ivar list<int> W: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last. This is the proposed sample.
        """

        acceptanceRatio = 1.0
        for comb in itertools.combinations(V, 2):

            #Check if comb[0] is ranked before comb[1] in V and W
            vIOverJ = 1
            wIOverJ = 1
            if V.index(comb[0]) > V.index(comb[1]):
                vIOverJ = 0
            if W.index(comb[0]) > W.index(comb[1]):
                wIOverJ = 0
        
            acceptanceRatio = acceptanceRatio * self.phi**(self.wmg[comb[0]][comb[1]]*(vIOverJ-wIOverJ))
        return acceptanceRatio

class MechanismMcmcSampleGeneratorMallowsAdjacentPairwiseFlip(MechanismMcmcSampleGeneratorMallows):

    def getNextSample(self, V):
        """
        Generate the next sample by randomly flipping two adjacent candidates.

        :ivar list<int> V: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last. This is the current sample.
        """

        # Select a random alternative in V to switch with its adacent alternatives.
        randPos = random.randint(0, len(V)-2)
        W = copy.deepcopy(V)
        d = V[randPos]
        c = V[randPos+1]
        W[randPos] = c
        W[randPos+1] = d

        # Check whether we should change to the new ranking.
        prMW = 1
        prMV = 1
        prob = min(1.0,(prMW/prMV)*pow(self.phi, self.wmg[d][c]))/2
        if random.random() <= prob:
            V = W
    
        return V

class MechanismMcmcSampleGeneratorMallowsRandShuffle(MechanismMcmcSampleGeneratorMallows):

    def __init__(self, wmg, phi, shuffleSize):
        self.wmg = wmg
        self.phi = phi
        self.shuffleSize = shuffleSize

    def getNextSample(self, V):
        """
        Generate the next sample by randomly shuffling candidates.

        :ivar list<int> V: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last. This is the current sample.
        """

        positions = range(0, len(self.wmg))
        randPoss = random.sample(positions, self.shuffleSize)
        flipSet = copy.deepcopy(randPoss)
        randPoss.sort()
        W = copy.deepcopy(V)
        for j in range(0, self.shuffleSize):
            W[randPoss[j]] = V[flipSet[j]]

        # Check whether we should change to the new ranking.
        prMW = 1.0
        prMV = 1.0
        acceptanceRatio = self.calcAcceptanceRatio(V, W)
        prob = min(1.0,(prMW/prMV)*acceptanceRatio)
        if random.random() <= prob:
            V = W
        return V

class MechanismMcmcSampleGeneratorMallowsJumpingDistribution(MechanismMcmcSampleGeneratorMallows):

    def getNextSample(self, V):
        """
        We generate a new ranking based on a Mallows-based jumping distribution. The algorithm is 
        described in "Bayesian Ordinal Peer Grading" by Raman and Joachims.

        :ivar list<int> V: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last.
        """

        phi = self.phi
        wmg = self.wmg
        W = []
        W.append(V[0])
        for j in range(2, len(V)+1):
            randomSelect = random.random()
            threshold = 0.0
            denom = 1.0
            for k in range(1, j):
                denom = denom + phi**k
            for k in range(1, j+1):
                numerator = phi**(j - k)
                threshold = threshold + numerator/denom
                if randomSelect <= threshold:
                    W.insert(k-1,V[j-1])
                    break    

        # Check whether we should change to the new ranking.
        acceptanceRatio = self.calcAcceptanceRatio(V, W)
        prob = min(1.0,acceptanceRatio)
        if random.random() <= prob:
            V = W

        return V

class MechanismMcmcSampleGeneratorMallowsPlakettLuce(MechanismMcmcSampleGeneratorMallows):

    def __init__(self, wmg, phi):
        self.wmg = wmg
        self.phi = phi
        self.plakettLuceProbs = self.calcDrawingProbs()

    def setWmg(self, wmg):
        """
        Function to set the wmg. When we change the wmg, we must also recalculate the plakett luce
        probabilities.

        ivar: dict<int,<dict,<int,int>>> wmg: A two-dimensional dictionary that associates integer
            representations of each pair of candidates, cand1 and cand2, with the number of times
            cand1 is ranked above cand2 minus the number of times cand2 is ranked above cand1. The
            dictionary represents a weighted majority graph for an election.
        """

        self.plakettLuceProbs = calcDrawingProbs()
        self.wmg = wmg

    def setPhi(self, phi):
        """
        Function to set phi. When we change the phi, we must also recalculate the plakett luce
        probabilities.

        :ivar float phi: A value for phi such that 0 <= phi <= 1.        
        """

        self.plakettLuceProbs = calcDrawingProbs()
        self.phi = phi

    def getNextSample(self, V):
        """
        Given a ranking over the candidates, generate a new ranking by assigning each candidate at
        position i a Plakett-Luce weight of phi^i and draw a new ranking.

        :ivar list<int> V: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last.
        """

        W, WProb = self.drawRankingPlakettLuce(V)
        VProb = self.calcProbOfVFromW(V, W)
        acceptanceRatio = self.calcAcceptanceRatio(V, W)
        prob = min(1.0, acceptanceRatio * (VProb/WProb))
        if random.random() <= prob:
            V = W
        return V

    def calcDrawingProbs(self):
        """
        Returns a vector that contains the probabily of an item being from each position. We say
        that every item in a order vector is drawn with weight phi^i where i is its position.
        """

        wmg = self.wmg
        phi = self.phi

        # We say the weight of the candidate in position i is phi^i.
        weights = []
        for i in range(0, len(wmg.keys())):
            weights.append(phi**i)

        # Calculate the probabilty that an item at each weight is drawn.
        totalWeight = sum(weights)
        for i in range(0, len(wmg.keys())):
            weights[i] = weights[i]/totalWeight

        return weights

    def drawRankingPlakettLuce(self, rankList):
        """
        Given an order vector over the candidates, draw candidates to generate a new order vector.

        :ivar list<int> rankList: Contains integer representations of each candidate in order of their
            rank in a vote, from first to last.
        """

        probs = self.plakettLuceProbs
        numCands = len(rankList)
        newRanking = []
        remainingCands = copy.deepcopy(rankList)
        probsCopy = copy.deepcopy(self.plakettLuceProbs)
        totalProb = sum(probs)

        # We will use prob to iteratively calculate the probabilty that we draw the order vector
        # that we end up drawing.
        prob = 1.0
        
        while (len(newRanking) < numCands):

            # We generate a random number from 0 to 1, and use it to select a candidate. 
            rand = random.random()
            threshold = 0.0
            for i in range(0, len(probsCopy)):
                threshold = threshold + probsCopy[i]/totalProb
                if rand <= threshold:
                    prob = prob * probsCopy[i]/totalProb
                    newRanking.append(remainingCands[i])
                    remainingCands.pop(i)
                    totalProb = totalProb - probsCopy[i]
                    probsCopy.pop(i)
                    break

        return newRanking, prob

    def calcProbOfVFromW(self, V, W):
        """
        Given a order vector V and an order vector W, calculate the probability that we generate
        V as our next sample if our current sample was W.

        :ivar list<int> V: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last.
        :ivar list<int> W: Contains integer representations of each candidate in order of their
            ranking in a vote, from first to last.
        """

        weights = range(0, len(V))
        i = 0
        for alt in W:
            weights[alt-1] = self.phi ** i
            i = i + 1
        
        # Calculate the probability that we draw V[0], V[1], and so on from W.
        prob = 1.0
        totalWeight = sum(weights)
        for alt in V:
            prob = prob * weights[alt-1]/totalWeight
            totalWeight = totalWeight - weights[alt-1]
    
        return prob

class MechanismMcmcSampleGeneratorCondorcet(MechanismMcmcSampleGenerator):

    def getNextSample(self, V):
        """
        Generate the next sample for the condorcet model. This algorithm is described in "Computing
        Optimal Bayesian Decisions for Rank Aggregation via MCMC Sampling," and is adapted from 
        code written by Lirong Xia.
        
        :ivar list<list<int> V: A two-dimensional list that for every pair of candidates cand1 and 
            cand2, V[cand1][cand2] contains 1 if cand1 is ranked above cand2 more times than cand2
            is ranked above cand1 and 0 otherwise.
        """

        cands = range(len(self.wmg))
        W = copy.deepcopy(V)
        allPairs = itertools.combinations(cands, 2)
        for pair in allPairs:
            a = pair[0]
            b = pair[1]
            if random.random() < 1.0/(1.0+pow(self.phi,self.wmg[a+1][b+1])):
                W[a][b] = 1
                W[b][a] = 0
            else:
                W[a][b] = 0
                W[b][a] = 1
        prMW = 1
        prMV = 1
        prob = min(1.0, prMW/prMV)
        if random.random() <= prob:
            V = W
        return V









