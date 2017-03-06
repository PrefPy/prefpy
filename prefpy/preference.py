"""
Author: Kevin J. Hwang
"""

class Preference():
    """
    The Preference class represents the preference of one or more voters. The underlying
    representation is a weighted majority graph.

    :ivar dict<dict<int,int>> wmgMap: A two-dimensional dictionary that associates each pair of 
        integer representations of candidates, cand1 and cand2, with the number of times cand1 is
        ranked above cand2 minus the number of times cand2 is ranked above cand1. This dictionary
        represents a weighted majority graph.
    :ivar int count: the number of voters holding this preference.
    """

    def __init__(self, wmgMap, count = 1):
        self.wmgMap = wmgMap
        self.count = count

    def isFullPreferenceOrder(self, candList):
        """
        Returns True if the underlying weighted majority graph contains a comparision between every
        pair of candidate and returns False otherwise.

        :ivar list<int> candList: Contains integer representations of each candidate.
        """

        # If a candidate is missing from the wmgMap or if there is a pair of candidates for which 
        # there is no value in the wmgMap, then the wmgMap cannot be a full preference order.
        for cand1 in candList:            
            if cand1 not in self.wmgMap.keys():
                return False
            for cand2 in candList:
                if cand1 == cand2:
                    continue
                if cand2 not in self.wmgMap[cand1].keys():
                    return False
        return True

    def containsTie(self):
        """
        Returns True if the underlying weighted majority graph contains a tie between any pair of
        candidates and returns False otherwise.
        """

        # If a value of 0 is present in the wmgMap, we assume that it represents a tie.
        for cand in self.wmgMap.keys():
            if 0 in self.wmgMap[cand].values():
                return True
        return False

    def getIncEdgesMap(self):
        """
        Returns a dictionary that associates numbers of incoming edges in the weighted majority
        graph with the candidates that have that number of incoming edges.
        """

        # We calculate the number of incoming edges for each candidate and store it into a dictionary 
        # that associates the number of incoming edges with the candidates with that number.
        incEdgesMap = dict()
        for cand1 in self.wmgMap.keys():
            incEdgesSum = 0
            for cand2 in self.wmgMap[cand1].keys():
                if self.wmgMap[cand1][cand2] > 0:
                    incEdgesSum += self.wmgMap[cand1][cand2]
            
            # Check if this is the first candidate associated with this number of associated edges.
            if incEdgesSum in incEdgesMap.keys():
                incEdgesMap[incEdgesSum].append(cand1)
            else:
                incEdgesMap[incEdgesSum] = [cand1]  

        return incEdgesMap

    def getRankMap(self):
        """
        Returns a dictionary that associates the integer representation of each candidate with its
        position in the ranking, starting from 1.
        """

        # We sort the candidates based on the number of incoming edges they have in the graph. If 
        # two candidates have the same number, we assume that they are tied.
        incEdgesMap = self.getIncEdgesMap()
        sortedKeys = sorted(incEdgesMap.keys(), reverse = True)
        rankMap = dict()
        pos = 1
        for key in sortedKeys:
            cands = incEdgesMap[key]
            for cand in cands:
                rankMap[cand] = pos
            pos += 1
        return rankMap

    def getReverseRankMap(self):
        """
        Returns a dictionary that associates each position in the ranking with a list of integer 
        representations of the candidates ranked at that position.
        """
        
        # We sort the candidates based on the number of incoming edges they have in the graph. If 
        # two candidates have the same number, we assume that they are tied.
        incEdgesMap = self.getIncEdgesMap()
        sortedKeys = sorted(incEdgesMap.keys(), reverse = True)
        reverseRankMap = dict()
        pos = 1
        for key in sortedKeys:
            cands = incEdgesMap[key]
            reverseRankMap[pos] = cands
            pos += 1
        return reverseRankMap

    def getOrderVector(self):
        """
        Returns a list of lists. Each list represents tiers of candidates. candidates in earlier
        tiers are preferred to candidates appearing in later tiers. Candidates in the same tier
        are preferred equally. 
        """

        # We sort the candidates based on the number of incoming edges they have in the graph. If 
        # two candidates have the same number, we assume that they are tied.
        incEdgesMap = self.getIncEdgesMap()
        sortedKeys = sorted(incEdgesMap.keys(), reverse = True)
        orderVector = []
        # print("sortedKeys",sortedKeys)
        # print("incEdgesMap", incEdgesMap)
        for key in sortedKeys:
            tier = []
            cands = incEdgesMap[key]
            # print("qq",cands)
            for cand in cands:
                tier.append(cand)
                # print("cand=",cand)
            # print("tier", tier)
            orderVector.append(tier[0])  # replace tier with tier[0]
        return orderVector