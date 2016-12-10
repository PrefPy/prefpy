# Filename: gmm_mixpl_moments.py
# Author: Peter Piech
# Date: 1/21/2016
# Description: Collection of moment condition value calculation
#              functions for Generalized Method of Moments algorithm
#              for mixtures of Plackett-Luce model rank data.

import numpy as np


def top2_reduced(votes):
    """
    Description:
        Top 2 alternatives 12 moment conditions values calculation
    Parameters:
        votes: ordinal preference data (numpy ndarray of integers)
    """
    res = np.zeros(12)
    for vote in votes:
        # the top ranked alternative is in vote[0][0], second in vote[1][0]
        if vote[0][0] == 0: # i.e. the first alt is ranked first
            res[0] += 1
            if vote[1][0] == 2:
                res[4] += 1
            elif vote[1][0] == 3:
                res[5] += 1
        elif vote[0][0] == 1:
            res[1] += 1
            if vote[1][0] == 0:
                res[6] += 1
            elif vote[1][0] == 3:
                res[7] += 1
        elif vote[0][0] == 2:
            res[2] += 1
            if vote[1][0] == 0:
                res[8] += 1
            elif vote[1][0] == 1:
                res[9] += 1
        elif vote[0][0] == 3:
            res[3] += 1
            if vote[1][0] == 1:
                res[10] += 1
            elif vote[1][0] == 2:
                res[11] += 1
    res /= len(votes)
    return res


def top2_full(votes):
    """
    Description:
        Top 2 alternatives 16 moment conditions values calculation
    Parameters:
        votes: ordinal preference data (numpy ndarray of integers)
    """
    res = np.zeros(16)
    for vote in votes:
        # the top ranked alternative is in vote[0][0], second in vote[1][0]
        if vote[0][0] == 0: # i.e. the first alt is ranked first
            res[0] += 1
            if vote[1][0] == 1: # i.e. the second alt is ranked second
                res[4] += 1
            elif vote[1][0] == 2:
                res[5] += 1
            elif vote[1][0] == 3:
                res[6] += 1
        elif vote[0][0] == 1:
            res[1] += 1
            if vote[1][0] == 0:
                res[7] += 1
            elif vote[1][0] == 2:
                res[8] += 1
            elif vote[1][0] == 3:
                res[9] += 1
        elif vote[0][0] == 2:
            res[2] += 1
            if vote[1][0] == 0:
                res[10] += 1
            elif vote[1][0] == 1:
                res[11] += 1
            elif vote[1][0] == 3:
                res[12] += 1
        elif vote[0][0] == 3:
            res[3] += 1
            if vote[1][0] == 0:
                res[13] += 1
            elif vote[1][0] == 1:
                res[14] += 1
            elif vote[1][0] == 2:
                res[15] += 1
    res /= len(votes)
    return res


def top3_reduced(votes):
    """
    Description:
        Top 3 alternatives 16 moment conditions values calculation
    Parameters:
        votes: ordinal preference data (numpy ndarray of integers)
    """
    res = np.zeros(16)
    for vote in votes:
        # the top ranked alternative is in vote[0][0], second in vote[1][0]
        if vote[0][0] == 0: # i.e. the first alt is ranked first
            res[0] += 1
            if vote[1][0] == 2:
                res[4] += 1
            elif vote[1][0] == 3:
                res[5] += 1
            elif vote[1][0] == 1 and vote[2][0] == 2:
                res[14] += 1
        elif vote[0][0] == 1:
            res[1] += 1
            if vote[1][0] == 0:
                res[6] += 1
            elif vote[1][0] == 3:
                res[7] += 1
            elif vote[1][0] == 2 and vote[2][0] == 3:
                res[15] += 1
        elif vote[0][0] == 2:
            res[2] += 1
            if vote[1][0] == 0:
                res[8] += 1
            elif vote[1][0] == 1:
                res[9] += 1
            elif vote[1][0] == 3 and vote[2][0] == 0:
                res[12] += 1
        elif vote[0][0] == 3:
            res[3] += 1
            if vote[1][0] == 1:
                res[10] += 1
            elif vote[1][0] == 2:
                res[11] += 1
            elif vote[1][0] == 0 and vote[2][0] == 1:
                res[13] += 1
    res /= len(votes)
    return res


def top3_full(votes):
    """
    Description:
        Top m - 1 alternatives q = m(m - 1) + 2m moment conditions values calculation
    Parameters:
        votes: ordinal preference data (numpy ndarray of integers)
    """
    #create array of zeros, length = q
    res = np.zeros(2 * len(votes[0]) + (len(votes[0]) * (len(votes[0]) - 1)))

    #iterate through each vote
    for vote in votes:
        #set verification boolean to true
        ver = True
        #check if vote belongs to c1 < c2 < c3, c2 < c3 < c1... moment
        for i in range(0, len(votes[0])):
            if vote[i][0] != vote[i - 1][0] + 1 and vote[i][0] != 0:
                ver = False
                break
        if ver:
            res[len(votes[0]) + (len(votes[0]) * (len(votes[0]) - 1)) + vote[0][0]] += 1

        #increment moment of top ranked choice ranked at the top
        res[vote[0][0]] += 1

        #top two moment
        add = 0
        if vote[0][0] > vote[1][0]:
            add = 1
        res[(vote[0][0] + 1) * (len(votes[0]) - 1) + add + vote[1][0]] += 1

    res /= len(votes) #normalize moments
    
    return res
