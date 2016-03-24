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
        # the top ranked alternative is in vote[0], second in vote[1]
        if vote[0] == 0: # i.e. the first alt is ranked first
            res[0] += 1
            if vote[1] == 2:
                res[4] += 1
            elif vote[1] == 3:
                res[5] += 1
        elif vote[0] == 1:
            res[1] += 1
            if vote[1] == 0:
                res[6] += 1
            elif vote[1] == 3:
                res[7] += 1
        elif vote[0] == 2:
            res[2] += 1
            if vote[1] == 0:
                res[8] += 1
            elif vote[1] == 1:
                res[9] += 1
        elif vote[0] == 3:
            res[3] += 1
            if vote[1] == 1:
                res[10] += 1
            elif vote[1] == 2:
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
        # the top ranked alternative is in vote[0], second in vote[1]
        if vote[0] == 0: # i.e. the first alt is ranked first
            res[0] += 1
            if vote[1] == 1: # i.e. the second alt is ranked second
                res[4] += 1
            elif vote[1] == 2:
                res[5] += 1
            elif vote[1] == 3:
                res[6] += 1
        elif vote[0] == 1:
            res[1] += 1
            if vote[1] == 0:
                res[7] += 1
            elif vote[1] == 2:
                res[8] += 1
            elif vote[1] == 3:
                res[9] += 1
        elif vote[0] == 2:
            res[2] += 1
            if vote[1] == 0:
                res[10] += 1
            elif vote[1] == 1:
                res[11] += 1
            elif vote[1] == 3:
                res[12] += 1
        elif vote[0] == 3:
            res[3] += 1
            if vote[1] == 0:
                res[13] += 1
            elif vote[1] == 1:
                res[14] += 1
            elif vote[1] == 2:
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
        # the top ranked alternative is in vote[0], second in vote[1]
        if vote[0] == 0: # i.e. the first alt is ranked first
            res[0] += 1
            if vote[1] == 2:
                res[4] += 1
            elif vote[1] == 3:
                res[5] += 1
            elif vote[1] == 1 and vote[2] == 2:
                res[14] += 1
        elif vote[0] == 1:
            res[1] += 1
            if vote[1] == 0:
                res[6] += 1
            elif vote[1] == 3:
                res[7] += 1
            elif vote[1] == 2 and vote[2] == 3:
                res[15] += 1
        elif vote[0] == 2:
            res[2] += 1
            if vote[1] == 0:
                res[8] += 1
            elif vote[1] == 1:
                res[9] += 1
            elif vote[1] == 3 and vote[2] == 0:
                res[12] += 1
        elif vote[0] == 3:
            res[3] += 1
            if vote[1] == 1:
                res[10] += 1
            elif vote[1] == 2:
                res[11] += 1
            elif vote[1] == 0 and vote[2] == 1:
                res[13] += 1
    res /= len(votes)
    return res


def top3_full(votes):
    """
    Description:
        Top 3 alternatives 20 moment conditions values calculation
    Parameters:
        votes: ordinal preference data (numpy ndarray of integers)
    """
    res = np.zeros(20)
    for vote in votes:
        # the top ranked alternative is in vote[0], second in vote[1]
        if vote[0] == 0: # i.e. the first alt is ranked first
            res[0] += 1
            if vote[1] == 1: # i.e. the second alt is ranked second
                res[4] += 1
                if vote[2] == 2:
                    res[18] += 1
            elif vote[1] == 2:
                res[5] += 1
            elif vote[1] == 3:
                res[6] += 1
        elif vote[0] == 1:
            res[1] += 1
            if vote[1] == 0:
                res[7] += 1
            elif vote[1] == 2:
                res[8] += 1
                if vote[2] == 3:
                    res[19] += 1
            elif vote[1] == 3:
                res[9] += 1
        elif vote[0] == 2:
            res[2] += 1
            if vote[1] == 0:
                res[10] += 1
            elif vote[1] == 1:
                res[11] += 1
            elif vote[1] == 3:
                res[12] += 1
                if vote[2] == 0:
                    res[16] += 1
        elif vote[0] == 3:
            res[3] += 1
            if vote[1] == 0:
                res[13] += 1
                if vote[2] == 1:
                    res[17] += 1
            elif vote[1] == 1:
                res[14] += 1
            elif vote[1] == 2:
                res[15] += 1
    res /= len(votes)
    return res
