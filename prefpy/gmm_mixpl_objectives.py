# Filename: gmm_mixpl_objectives.py
# Author: Peter Piech
# Date: 1/21/2016
# Description: Collection of objective functions for
#              Generalized Method of Moments algorithm
#              for mixtures of two Plackett-Luce models
#              rank data over four alternatives.

import numpy as np


def uncons_term(params, c):
    """
    Description:
        Computes an additional value for the objective function value
        when used in an unconstrained optimization formulation.
    Parameters:
        params: all parameters for the Plackett-Luce mixture model (numpy ndarray)
        c:      constant multiplier scaling factor of the returned term
    """
    return (c * ((np.sum(params[1:5]) - 1)**2)) + (c * ((np.sum(params[5:]) - 1)**2))

def top2_reduced(params, moments):
    """
    Description:
        Top 2 alternatives 12 moment conditions objective function
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    params = np.asarray(params)
    alpha = params[0]
    a = params[1:5]
    b = params[5:]
    p = np.asarray(moments)
    p1 = alpha*a+(1-alpha)*b-p[:4]
    p21 = alpha*a[0]*a[2:]/(1-a[0])+(1-alpha)*b[0]*b[2:]/(1-b[0])-p[4:6]
    p22 = alpha*a[1]*np.hstack((a[0],a[3]))/(1-a[1])+(1-alpha)*b[1]*np.hstack((b[0],b[3]))/(1-b[1])-p[6:8]
    p23 = alpha*a[2]*a[:2]/(1-a[2])+(1-alpha)*b[2]*b[:2]/(1-b[2])-p[8:10]
    p24 = alpha*a[3]*a[1:3]/(1-a[3])+(1-alpha)*b[3]*b[1:3]/(1-b[3])-p[10:]
    allp = np.concatenate((p1,p21,p22,p23,p24))
    return np.sum(allp**2)

def top2_reduced_unconstrained(params, moments):
    """
    Description:
        Top 2 alternatives 12 moment conditions objective function for use
        with an unconstrained formulation of the optimization. Adds an
        additional term to the function value multiplied by a very large
        constant.
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    return top2_reduced(params, moments) + uncons_term(params, c=1e6)

def top2_full(params, moments):
    """
    Description:
        Top 2 alternatives 16 moment conditions objective function
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    params = np.asarray(params)
    alpha = params[0]
    a = params[1:5]
    b = params[5:]
    p = np.asarray(moments)
    p1 = alpha*a+(1-alpha)*b-p[:4]
    p21 = alpha*a[0]*a[1:]/(1-a[0])+(1-alpha)*b[0]*b[1:]/(1-b[0])-p[4:7]
    p22 = alpha*a[1]*np.hstack((a[0],a[2:]))/(1-a[1])+(1-alpha)*b[1]*np.hstack((b[0],b[2:]))/(1-b[1])-p[7:10]
    p23 = alpha*a[2]*np.hstack((a[:2],a[3]))/(1-a[2])+(1-alpha)*b[2]*np.hstack((b[:2],b[3]))/(1-b[2])-p[10:13]
    p24 = alpha*a[3]*a[:3]/(1-a[3])+(1-alpha)*b[3]*b[:3]/(1-b[3])-p[13:]
    allp = np.concatenate((p1,p21,p22,p23,p24))
    return np.sum(allp**2)

def top2_full_unconstrained(params, moments):
    """
    Description:
        Top 2 alternatives 16 moment conditions objective function for use
        with an unconstrained formulation of the optimization. Adds an
        additional term to the function value multiplied by a very large
        constant.
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    return top2_full(params, moments) + uncons_term(params, c=1e6)

def top3_reduced(params, moments):
    """
    Description:
        Top 3 alternatives 16 moment conditions objective function
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    params = np.asarray(params)
    alpha = params[0]
    a = params[1:5]
    b = params[5:]
    p = np.asarray(moments)
    p1 = alpha*a+(1-alpha)*b-p[:4]
    p21 = alpha*a[0]*a[2:]/(1-a[0])+(1-alpha)*b[0]*b[2:]/(1-b[0])-p[4:6]
    p22 = alpha*a[1]*np.hstack((a[0],a[3]))/(1-a[1])+(1-alpha)*b[1]*np.hstack((b[0],b[3]))/(1-b[1])-p[6:8]
    p23 = alpha*a[2]*a[:2]/(1-a[2])+(1-alpha)*b[2]*b[:2]/(1-b[2])-p[8:10]
    p24 = alpha*a[3]*a[1:3]/(1-a[3])+(1-alpha)*b[3]*b[1:3]/(1-b[3])-p[10:12]
    p3 = np.array([
        alpha*a[0]*a[2]*a[3]/(1-a[2])/(a[0]+a[1])+(1-alpha)*b[0]*b[2]*b[3]/(1-b[2])/(b[0]+b[1])-p[12],
        alpha*a[0]*a[1]*a[3]/(1-a[3])/(a[1]+a[2])+(1-alpha)*b[0]*b[1]*b[3]/(1-b[3])/(b[1]+b[2])-p[13],
        alpha*a[0]*a[1]*a[2]/(1-a[0])/(a[3]+a[2])+(1-alpha)*b[0]*b[1]*b[2]/(1-b[0])/(b[3]+b[2])-p[14],
        alpha*a[2]*a[1]*a[3]/(1-a[1])/(a[0]+a[3])+(1-alpha)*b[2]*b[1]*b[3]/(1-b[1])/(b[0]+b[3])-p[15]
        ])
    allp = np.concatenate((p1,p21,p22,p23,p24,p3))
    return np.sum(allp**2)

def top3_reduced_unconstrained(params, moments):
    """
    Description:
        Top 3 alternatives 16 moment conditions objective function for use
        with an unconstrained formulation of the optimization. Adds an
        additional term to the function value multiplied by a very large
        constant.
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    return top3_reduced(params, moments) + uncons_term(params, c=1e6)

def top3_full(params, moments):
    """
    Description:
        Top 3 alternatives 20 moment conditions objective function
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    params = np.asarray(params)
    alpha = params[0]
    a = params[1:5]
    b = params[5:]
    p = np.asarray(moments)
    p1 = alpha*a+(1-alpha)*b-p[:4]
    p21 = alpha*a[0]*a[1:]/(1-a[0])+(1-alpha)*b[0]*b[1:]/(1-b[0])-p[4:7]
    p22 = alpha*a[1]*np.hstack((a[0],a[2:]))/(1-a[1])+(1-alpha)*b[1]*np.hstack((b[0],b[2:]))/(1-b[1])-p[7:10]
    p23 = alpha*a[2]*np.hstack((a[:2],a[3]))/(1-a[2])+(1-alpha)*b[2]*np.hstack((b[:2],b[3]))/(1-b[2])-p[10:13]
    p24 = alpha*a[3]*a[:3]/(1-a[3])+(1-alpha)*b[3]*b[:3]/(1-b[3])-p[13:16]
    p3 = np.array([
        alpha*a[0]*a[2]*a[3]/(1-a[2])/(a[0]+a[1])+(1-alpha)*b[0]*b[2]*b[3]/(1-b[2])/(b[0]+b[1])-p[16],
        alpha*a[0]*a[1]*a[3]/(1-a[3])/(a[1]+a[2])+(1-alpha)*b[0]*b[1]*b[3]/(1-b[3])/(b[1]+b[2])-p[17],
        alpha*a[0]*a[1]*a[2]/(1-a[0])/(a[3]+a[2])+(1-alpha)*b[0]*b[1]*b[2]/(1-b[0])/(b[3]+b[2])-p[18],
        alpha*a[2]*a[1]*a[3]/(1-a[1])/(a[0]+a[3])+(1-alpha)*b[2]*b[1]*b[3]/(1-b[1])/(b[0]+b[3])-p[19]
        ])
    allp = np.concatenate((p1,p21,p22,p23,p24,p3))
    return np.sum(allp**2)

def top3_full_unconstrained(params, moments):
    """
    Description:
        Top 3 alternatives 20 moment conditions objective function for use
        with an unconstrained formulation of the optimization. Adds an
        additional term to the function value multiplied by a very large
        constant.
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    return top3_full(params, moments) + uncons_term(params, c=1e6)
