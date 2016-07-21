import numpy
import copy

class UtilityFunction():
    """
    The parent class for all utility functions. All classes that inherit from this class must
    implement the method getUtilities(), which returns a list of the utilities given a decision,
    consisting of one or more candidates, and a ranking over the candidates, and must set isLoss
    to either True or False, which identifies the utility function as a loss function or a gain
    function.
    """

    def getUtility(self, decision, sample, aggregationMode = "avg"):
        """
        Get the utility of a given decision given a preference. 

        :ivar list<int> decision: Contains a list of integer representations of candidates in the 
            current decision.
        :ivar sample: A representation of a preference. We do not assume that it is of a certain 
            type here and merely pass it to the getUtilities() method.
        ivar str aggregationMode: Identifies the aggregation mode of the utility function when the
            decision selects more than one candidate. If the mode is "avg," the utility will be the
            averge of that of each candidate. If "min," the utility will be the minimum, and if 
            "max," the utility will xbe the maximum. By default the aggregation mode will be "avg."
        """

        utilities = self.getUtilities(decision, sample)
        if aggregationMode == "avg":
            utility = numpy.mean(utilities)
        elif aggregationMode == "min":
            utility = min(utilities)
        elif aggregationMode == "max":
            utility = max(utilities)
        else:
            print("ERROR: aggregation mode not recognized")
            exit()
        return utility

class UtilityFunctionMallowsPosScoring(UtilityFunction):
    """
    The positional scoring utility function for the Mallows model. By default, this will be 
    constructed as a loss function. 
    """

    def __init__(self, scoringVector, isLoss = True):
        self.scoringVector = scoringVector
        self.isLoss = True

    def getScoringVector(self, orderVector):
        """
        Returns the scoring vector. This function is called by getUtilities()

        :ivar list<int> orderVector: A list of integer representations for each candidate ordered
            from most preferred to least.
        """

        return self.scoringVector

    def getUtilities(self, decision, orderVector):
        """
        Returns a floats that contains the utilities of every candidate in the decision.

        :ivar list<int> decision: Contains a list of integer representations of candidates in the 
            current decision.
        :ivar list<int> orderVector: A list of integer representations for each candidate ordered
            from most preferred to least.
        """
        
        scoringVector = self.getScoringVector(orderVector)
        utilities = []
        for alt in decision:
            altPosition = orderVector.index(alt)
            utility = float(scoringVector[altPosition])
            if self.isLoss == True:
                utility = -1*utility
            utilities.append(utility)
        return utilities

class UtilityFunctionMallowsTopK(UtilityFunctionMallowsPosScoring):
    """
    The top-k utility function for the Mallows model. By default, this will be constructed as a 
    loss function.
    """
    
    def __init__(self, k, isLoss=True):
        self.k = k
        self.isLoss = True

    def getScoringVector(self, orderVector):
        """
        Returns a scoring vector such that the first k candidates recieve 1 point and all others 
        recive 0  This function is called by getUtilities() which is implemented in the parent
        class.

        :ivar list<int> orderVector: A list of integer representations for each candidate ordered
            from most preferred to least.
        """

        scoringVector = []
        for i in range(0, self.k):
            scoringVector.append(1)
        for i in range(self.k, len(orderVector)):
            scoringVector.append(0)
        return scoringVector

class UtilityFunctionMallowsZeroOne(UtilityFunctionMallowsPosScoring):
    """
    The zero-one utility function for the Mallows model. By default, this will be constructed as a
    loss function.
    """

    def __init__(self, isLoss = True):
        self.isLoss = isLoss

    def getScoringVector(self, orderVector):
        """
        Returns the scoring vector [1,0,0,...,0]. This function is called by getUtilities() 
        which is implemented in the parent class.

        :ivar list<int> orderVector: A list of integer representations for each candidate ordered
            from most preferred to least.
        """
        
        scoringVector = []
        scoringVector.append(1)
        for i in range(1, len(orderVector)):
            scoringVector.append(0)
        return scoringVector

class UtilityFunctionCondorcetTopK(UtilityFunction):
    """
    The top-k utility function for the Condorcet model. By default, this will be constructed as a
    gain function.
    """

    def __init__(self, k, isLoss = False):
        self.k = k
        self.isLoss = False

    def getUtilities(self, decision, binaryRelations):
        """
        Returns a floats that contains the utilities of every candidate in the decision. This was 
        adapted from code written by Lirong Xia.

        :ivar list<int> decision: Contains a list of integer representations of candidates in the 
            current decision.
        :ivar list<list,int> binaryRelations: A two-dimensional array whose number of rows and 
            colums is equal to the number of candidates. For each pair of candidates, cand1 and
            cand2, binaryRelations[cand1-1][cand2-1] contains 1 if cand1 is ranked above cand2
            and 0 otherwise.
        """

        m = len(binaryRelations)
        utilities = []
        for cand in decision:
            tops = [cand-1]
            index = 0
            while index < len(tops):
                s = tops[index]
                for j in range(m):
                    if j == s:
                        continue
                    if binaryRelations[j][s] > 0:
                        if j not in tops:
                            tops.append(j)
                index += 1
            if len(tops) <= self.k:
                if self.isLoss == False:
                    utilities.append(1.0)
                elif self.isLoss == True:
                    utilities.append(-1.0)
            else:
                utilities.append(0.0)
        return utilities
