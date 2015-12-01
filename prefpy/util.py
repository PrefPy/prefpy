# Filename: util.py
# Author: Peter Piech
# Date: 10/4/2015
# Description: Miscellaneous utility functions

def get_index_nested(x, i):
    """
    Description:
        Returns the first index of the array (vector) x containing the value i.
    Parameters:
        x: one-dimensional array
        i: search value
    """
    for ind in range(len(x)):
        if i == x[ind]:
            return ind
    return -1 # not found
