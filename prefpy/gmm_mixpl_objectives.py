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
        Top m - 1 alternatives m(m - 1) + 2m moment conditions objective function
    Parameters:
        params:  all parameters for the Plackett-Luce mixture model (numpy ndarray)
        moments: values of the moment conditions from the data (numpy ndarray)
    """
    #variables
    params = np.asarray(params) #convert numpy matrix to list
    alpha = params[0] #first parameter is the alpha value
    half = int((len(params) - 1) / 2) #assuming 2 mixtures
    a = params[1:half + 1] #first mixture
    b = params[half + 1:] #second mixture
    p = np.asarray(moments) #convert numpy matrix to list
    p1 = list(alpha*a+(1-alpha)*b-p[:half]) #new list with one element
    p2 = [] #new empty list

    #iterate through each 
    for i in range(0, half):
        #alpha times the score of a given point in mixture one, mutiplied by
            #each of the other scores, divided by the sum of the other values
        #Each of these top two plackett-luce values is added to the same values
            #from the other mixture, then the moment value is subtracted for those
            #top two from the vote
        p1 += list(alpha*a[i]*np.hstack((a[:i],a[i + 1:]))/(1-a[i])
            +(1-alpha)*b[i]*np.hstack((b[:i],b[i + 1:]))/(1-b[i])
            -p[half + (half - 1) * i:half + (half - 1) * (i + 1)])

    #iterate through each value in each mixture
    for i in range(0, half):
        #begin with alpha values for given mixture
        num_a = alpha
        num_b = 1 - alpha

        #iterate again
        for j in range(0, half):
            #this eventually multiplies all values to its alpha
            num_a *= a[j]
            num_b *= b[j]
            #divide by the sum of other values
            if j > i:
                num_a /= np.sum(np.concatenate((a[j:], a[:i])))
                num_b /= np.sum(np.concatenate((b[j:], b[:i])))
            elif j < i:
                num_a /= np.sum(a[j:i])
                num_b /= np.sum(b[j:i])
        p2.append(num_a + num_b - p[half + (half * (half - 1)) + i])
    p3 = np.array(p2)
    #create one array
    allp = np.concatenate((p1,p3))
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
