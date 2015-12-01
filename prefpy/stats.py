# Filename: statistics.py
# Author: Peter Piech
# Date: 10/4/2015
# Description: Module for computing statistical results

import numpy as np


def mse(mean, estimator):
    """
    Description:
        Calculates the Mean Squared Error (MSE) of
        an estimation on flat arrays.
    Parameters:
        mean:      actual value
        estimator: estimated value of the mean
    """
    return np.mean((estimator - mean) ** 2, axis=0)
