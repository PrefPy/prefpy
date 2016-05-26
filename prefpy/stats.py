# Filename: statistics.py
# Author: Peter Piech
# Date: 10/4/2015
# Description: Module for computing statistical results


###################### NOTICE: ######################
# The current names of the functions are misleading
# and represent an incorrect understanding of the
# statistical measures.  All functions below operate
# on a single data point and can be later aggregated
# using a whole set of data points in the appropriate
# fashion.  The names of the functions are embedded
# in many files within this package and will be
# renamed correctly in the future once time allows
# for this massive endeavor.
#####################################################

import numpy as np


def mse(mean, estimator):
    """
    Description:
        Calculates the Mean Squared Error (MSE) of
        an estimation on flat numpy ndarrays.
    Parameters:
        mean:      actual value (numpy ndarray)
        estimator: estimated value of the mean (numpy ndarray)
    """
    return np.mean((np.asarray(estimator) - np.asarray(mean)) ** 2, axis=0)

def sse(mean, estimator):
    """
    Description:
        Calculates the Sum of Squared Errors (SSE) of
        an estimation on flat numpy ndarrays.
    Parameters:
        mean:      actual value (numpy ndarray)
        estimator: estimated value of the mean (numpy ndarray)
    """
    return np.sum((np.asarray(estimator) - np.asarray(mean)) ** 2, axis=0)

#deprecated
def mix2PL_mse(mean, estimator, m):
    """
    Description:
        Calculates the Mean Squared Error (MSE) of an
        estimator of a mixture of 2 Plackett-Luce models,
        on flat numpy ndarrays, where the first element is
        the mixing proportion of the first model defined
        as the minimum MSE over the inverse permutations of
        the estimator.
    Parameters:
        mean:      actual value (numpy ndarray)
        estimator: estimated value of the mean (numpy ndarray)
        m:         number of alternatives in each of the two models
    """
    mse1 = mse(mean, estimator)
    estimator = np.hstack((1 - estimator[0], estimator[m+1:], estimator[1:m+1]))
    mse2 = mse(mean, estimator)
    return min(mse1, mse2)

#deprecated
def mix2PL_sse(mean, estimator, m):
    """
    Description:
        Calculates the Sum of Squared Errors (SSE) of an
        estimator of a mixture of 2 Plackett-Luce models,
        on flat numpy ndarrays, where the first element is
        the mixing proportion of the first model defined
        as the minimum SSE over the inverse permutations of
        the estimator.
    Parameters:
        mean:      actual value (numpy ndarray)
        estimator: estimated value of the mean (numpy ndarray)
        m:         number of alternatives in each of the two models
    """
    sse1 = sse(mean, estimator)
    estimator = np.hstack((1 - estimator[0], estimator[m+1:], estimator[1:m+1]))
    sse2 = sse(mean, estimator)
    return min(sse1, sse2)

#deprecated
def mix2PL_wsse(mean, estimator, m):
    """
    Description:
        Calculates the weighted Sum of Squared Errors (WSSE)
        of an estimator of a mixture of 2 Plackett-Luce models,
        on flat numpy ndarrays, where the first element is
        the mixing proportion of the first model defined
        as the minimum WSSE over the inverse permutations of
        the estimator.
    Parameters:
        mean:      actual value (numpy ndarray)
        estimator: estimated value of the mean (numpy ndarray)
        m:         number of alternatives in each of the two models
    """
    def wsse(mean1, est1, m1):
        return (((est1[0] - mean1[0])**2) +
                (mean1[0]*np.sum((np.asarray(est1[1:m1+1]) - np.asarray(mean1[1:m1+1]))**2)) +
                ((1 - mean1[0]) * np.sum((np.asarray(est1[m1+1:]) - np.asarray(mean1[m1+1:]))**2))
               )
    wsse1 = wsse(mean, estimator, m)
    estimator = np.hstack((1 - estimator[0], estimator[m+1:], estimator[1:m+1]))
    wsse2 = wsse(mean, estimator, m)
    return min(wsse1, wsse2)
