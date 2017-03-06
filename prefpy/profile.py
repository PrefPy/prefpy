"""
Author: Kevin J. Hwang
"""
import copy
from . import prefpy_io
import itertools
import math
import json
import os
from .preference import Preference
# import preference

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

    #----------------------------------------------------------------------------------------------

    def genWmgMapFromRankMap(self, rankMap):
        """
        Converts a single rankMap into a weighted majorty graph (wmg). We return the wmg as a 
        two-dimensional dictionary that associates integer representations of each pair of candidates,
        cand1 and cand2, with the number of times cand1 is ranked above cand2 minus the number of times
        cand2 is ranked above cand1. This is called by importPreflibFile().

        :ivar list<int> candList: Contains integer representations of each candidate.
        :ivar dict<int,int> rankMap: Associates integer representations of each candidate with its
            ranking in a single vote.
        """

        wmgMap = dict()
        for cand1, cand2 in itertools.combinations(rankMap.keys(), 2):

            # Check whether or not the candidates are already present in the dictionary.
            if cand1 not in wmgMap.keys():
                wmgMap[cand1] = dict()
            if cand2 not in wmgMap.keys():
                wmgMap[cand2] = dict()
                
            # Check which candidate is ranked above the other. Then assign 1 or -1 as appropriate.       
            if rankMap[cand1] < rankMap[cand2]:
                wmgMap[cand1][cand2] = 1
                wmgMap[cand2][cand1] = -1 
            elif rankMap[cand1] > rankMap[cand2]:
                wmgMap[cand1][cand2] = -1
                wmgMap[cand2][cand1] = 1

            # If the two candidates are tied, We make 0 the number of edges between them.
            elif rankMap[cand1] == rankMap[cand2]:
                wmgMap[cand1][cand2] = 0
                wmgMap[cand2][cand1] = 0

        return wmgMap

    def exportPreflibFile(self, fileName):
        """
        Exports a preflib format file that contains all the information of the current Profile.

        :ivar str fileName: The name of the output file to be exported.
        """

        elecType = self.getElecType()

        if elecType != "soc" and elecType != "toc" and elecType != "soi" and elecType != "toi":
            print("ERROR: printing current type to preflib format is not supported")
            exit()

        # Generate a list of reverse rankMaps, one for each vote. This will allow us to easiliy
        # identify ties.
        reverseRankMaps = self.getReverseRankMaps()

        outfileObj = open(fileName, 'w')

        # Print the number of candidates and the integer representation and name of each candidate.
        outfileObj.write(str(self.numCands))
        for candInt, cand in self.candMap.items():
            outfileObj.write("\n" + str(candInt) + "," + cand)

        # Sum up the number of preferences that are represented.
        preferenceCount = 0
        for preference in self.preferences:
            preferenceCount += preference.count

        # Print the number of voters, the sum of vote count, and the number of unique orders.
        outfileObj.write("\n" + str(self.numVoters) + "," + str(preferenceCount) + "," + str(len(self.preferences)))

        for i in range(0, len(reverseRankMaps)):

            # First, print the number of times the preference appears.
            outfileObj.write("\n" + str(self.preferences[i].count))
            
            reverseRankMap = reverseRankMaps[i]

            # We sort the positions in increasing order and print the candidates at each position
            # in order.
            sortedKeys = sorted(reverseRankMap.keys())
            for key in sortedKeys:

                cands = reverseRankMap[key]

                # If only one candidate is in a particular position, we assume there is no tie.
                if len(cands) == 1:
                    outfileObj.write("," + str(cands[0]))

                # If more than one candidate is in a particular position, they are tied. We print
                # brackets around the candidates.
                elif len(cands) > 1:
                    outfileObj.write(",{" + str(cands[0]))
                    for j in range(1, len(cands)):
                        outfileObj.write("," + str(cands[j]))
                    outfileObj.write("}")
                    
        outfileObj.close()            

    def importPreflibFile(self, fileName):
        """
        Imports a preflib format file that contains all the information of a Profile. This function
        will completely override all members of the current Profile object. Currently, we assume 
        that in an election where incomplete ordering are allowed, if a voter ranks only one 
        candidate, then the voter did not prefer any candidates over another. This may lead to some
        discrepancies when importing and exporting a .toi preflib file or a .soi preflib file.

        :ivar str fileName: The name of the input file to be imported.
        """

        # Use the functionality found in io to read the file.
        elecFileObj = open(fileName, 'r')
        self.candMap, rankMaps, wmgMapsCounts, self.numVoters = prefpy_io.read_election_file(elecFileObj)
        elecFileObj.close()

        self.numCands = len(self.candMap.keys())

        # Go through the rankMaps and generate a wmgMap for each vote. Use the wmgMap to create a
        # Preference object.
        self.preferences = []
        for i in range(0, len(rankMaps)):
            wmgMap = self.genWmgMapFromRankMap(rankMaps[i])
            self.preferences.append(Preference(wmgMap, wmgMapsCounts[i]))

    def exportJsonFile(self, fileName):
        """
        Exports a json file that contains all the information of the current Profile.

        :ivar str fileName: The name of the output file to be exported.
        """

        # Because our Profile class is not directly JSON serializable, we exporrt the underlying 
        # dictionary. 
        data = dict()
        for key in self.__dict__.keys():
            if key != "preferences":
                data[key] = self.__dict__[key]
        
        # The Preference class is also not directly JSON serializable, so we export the underlying
        # dictionary for each Preference object.
        preferenceDicts = []
        for preference in self.preferences:
            preferenceDict = dict()
            for key in preference.__dict__.keys():
                preferenceDict[key] = preference.__dict__[key]
            preferenceDicts.append(preferenceDict)
        data["preferences"] = preferenceDicts

        outfile = open(fileName, 'w')
        json.dump(data, outfile)
        outfile.close()

    def importJsonFile(self, fileName):
        """
        Imports a json file that contains all the information of a Profile. This function will
        completely override all members of the current Profile object.

        :ivar str fileName: The name of the input file to be imported.
        """

        infile = open(fileName)
        data = json.load(infile)
        infile.close()
        
        self.numCands = int(data["numCands"])
        self.numVoters = int(data["numVoters"])

        # Because the json.load function imports everything as unicode strings, we will go through
        # the candMap dictionary and convert all the keys to integers and convert all the values to
        # ascii strings.
        candMap = dict()
        for key in data["candMap"].keys():
            candMap[int(key)] = data["candMap"][key].encode("ascii")
        self.candMap = candMap
        
        # The Preference class is also not directly JSON serializable, so we exported the 
        # underlying dictionary for each Preference object. When we import, we will create a 
        # Preference object from these dictionaries.
        self.preferences = []
        for preferenceMap in data["preferences"]:
            count = int(preferenceMap["count"])

            # Because json.load imports all the items in the wmgMap as unicode strings, we need to
            # convert all the keys and values into integers.
            preferenceWmgMap = preferenceMap["wmgMap"]
            wmgMap = dict()
            for key in preferenceWmgMap.keys():
                wmgMap[int(key)] = dict()
                for key2 in preferenceWmgMap[key].keys():
                    wmgMap[int(key)][int(key2)] = int(preferenceWmgMap[key][key2])

            self.preferences.append(Preference(wmgMap, count))

    #----------------------------------------------------------------------------------------------
