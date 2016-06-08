"""
Author: Kevin J. Hwang
"""
import copy
import io
import itertools
import math
from preference import Preference

class Profile():
    """
    The Profile class is the representation of an election Profile.
    
    :ivar dict<int,str> candMap: Associates integer representations of each candidate with the name
        of the candidate.
    :ivar int numCands: The number of candidates in the election.
    :ivar list<Preference> preferences: Contains objects that represent preferences held over the
        candidates by individual voters.
    :ivar int numVoters: The number of voters in the election.   
    """

    def __init__(self, candMap, preferences):

        self.candMap = candMap
        self.numCands = len(candMap.keys())
        self.preferences = preferences
        self.numVoters = 0
        for preference in preferences:
            self.numVoters += preference.count

    def getElecType(self): 
        """
        Determines whether the list of Preference objects represents complete strict orderings over
        the candidates (soc), incomplete strict orderings (soi), complete orderings with ties (toc), 
        or incomplete orderings with ties (toi). WARNING: elections that do not fall under the 
        above four categories may be falsely identified.
        """

        tiesPresent = False
        incompletePresent = False

        # Go through the list of Preferences and see if any contain a tie or are incomplete.
        for preference in self.preferences:
            if preference.containsTie() == True:
                tiesPresent = True
                break
            if preference.isFullPreferenceOrder(self.candMap.keys()) == False:
                incompletePresent = True
                break

        if tiesPresent == False and incompletePresent == False:
            elecType = "soc"
        elif tiesPresent == False and incompletePresent == True:
            elecType = "soi"
        elif tiesPresent == True and incompletePresent == False:
            elecType = "toc"
        elif tiesPresent == True and incompletePresent == True:
            elecType = "toi"
        return elecType

    def getPreferenceCounts(self):
        """
        Returns a list of the number of times each preference is given.
        """

        preferenceCounts = []
        for preference in self.preferences:
            preferenceCounts.append(preference.count)
        return preferenceCounts

    def getRankMaps(self):
        """
        Returns a list of dictionaries, one for each preference, that associates the integer 
        representation of each candidate with its position in the ranking, starting from 1 and
        returns a list of the number of times each preference is given.
        """

        rankMaps = []
        for preference in self.preferences:
            rankMaps.append(preference.getRankMap())
        return rankMaps
    def getReverseRankMaps(self):
        """
        Returns a list of dictionaries, one for each preference, that associates each position in
        the ranking with a list of integer representations of the candidates ranked at that 
        position and returns a list of the number of times each preference is given.
        """

        reverseRankMaps = []
        for preference in self.preferences:
            reverseRankMaps.append(preference.getReverseRankMap())
        return reverseRankMaps

    def getOrderVectors(self):
        """
        Returns a list of lists, one for each preference, of candidates ordered from most preferred
        to least. Note that ties are not indicated in the returned lists. Also returns a list of
        the number of times each preference is given.
        """

        orderVectors = []
        for preference in self.preferences:
            orderVectors.append(preference.getOrderVector())
        return orderVectors

    def getWmg(self, normalize = False):
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
        for cand in self.candMap.keys():
            wmgMap[cand] = dict()
        for cand1, cand2 in itertools.combinations(self.candMap.keys(), 2):
            wmgMap[cand1][cand2] = 0
            wmgMap[cand2][cand1] = 0

        # Go through the wmgMaps and increment the value of each edge in our final graph with the 
        # edges in each of the wmgMaps. We take into account the number of times that the vote 
        # occured.
        for i in range(0, len(self.preferences)):
            preference = self.preferences[i]
            preferenceWmgMap = preference.wmgMap
            for cand1, cand2 in itertools.combinations(preferenceWmgMap.keys(), 2):
                if cand2 in preferenceWmgMap[cand1].keys():
                    wmgMap[cand1][cand2] += preferenceWmgMap[cand1][cand2]*preference.count
                    wmgMap[cand2][cand1] += preferenceWmgMap[cand2][cand1]*preference.count

        # By default, we assume that the weighted majority graph should not be normalized. If
        # desired, we normalize by dividing each edge by the value of the largest edge. 
        if (normalize == True):
            maxEdge = float('-inf')
            for cand in wmgMap.keys():
                maxEdge = max(maxEdge, max(wmgMap[cand].values()))
            for cand1 in wmgMap.keys():
                for cand2 in wmgMap[cand1].keys():
                    wmgMap[cand1][cand2] = float(wmgMap[cand1][cand2])/maxEdge
        
        return wmgMap