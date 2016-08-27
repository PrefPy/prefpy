from scipy.stats import norm, rankdata
import time
import math
import numpy as np
import pytz
import pandas as pd
from numpy.linalg import inv, pinv

#probability of one alternative is preferred to another.
#x is the difference between the means of the two alternatives
def prPair(x):
    return norm.cdf(x, 0, scale=math.sqrt(2))

#derivative of probability
def dPrPair(x):
    return np.exp(-x ** 2/4)/(2 * math.sqrt(math.pi))

#second order derivative of probability
def ddPrPair(x):
    return -x * np.exp(- x ** 2/4)/(4 * math.sqrt(math.pi))

#calculate the breakings
def dataBreaking(data, m, normalize):
    breaking = np.zeros((m, m), int)
    n = len(data)
    for d in data:
        for i in range(0, m):
            for j in range(i+1, m):
                breaking[d[i], d[j]] += 1
    if normalize:
        return breaking/n
    else:
        return breaking

#for debugging purpose
def trueBreaking(Mu):
    m=len(Mu)
    A=np.zeros((m,m), float)

    for i in range(0,m):
        for j in range(0,m):
            x = Mu[i] - Mu[j]
            A[i,j] = norm.cdf(x, 0, scale=math.sqrt(2))
    np.fill_diagonal(A,0)
    return A

#To calculate the gradient of objective function
def gradientPrPair(breaking, theta, n):
    theta = theta[0]
    m = len(theta)
    grad = np.zeros((m, 1), float)
    for i in range(0, m):
        for j in range(0, m):
            if j != i:
                x = theta[i] - theta[j]
                grad[i] += 2 * (breaking[i][j] - n * prPair(x)) * dPrPair(x)
    return grad

#To calculate the Hessian matrix
def hessianPrPair(breaking, theta, n):
    theta = theta[0]
    m = len(theta)
    hessian = np.zeros((m, m), float)
    for i in range(0, m):
        for j in range(0, m):
            if j != i:
                x = theta[i] - theta[j]
                hessian[i][i] += 2*(breaking[i][j]-n*prPair(x))*ddPrPair(x) - 2*n*(dPrPair(x))**2
                if j > i:
                    hessian[i][j] = 2*n*dPrPair(x)**2-2*ddPrPair(breaking[i][j]-n*prPair(x))
                else:
                    hessian[i][j] = hessian[j][i]
    return hessian

#' GMM Method for Estimating Random Utility Model wih Normal dsitributions
#'
#' @param Data.pairs data broken up into pairs
#' @param m number of alternatives
#' @param itr number of itrations to run
#' @param Var indicator for difference variance (default is FALSE)
#' @param prior magnitude of fake observations input into the model
#' @return Estimated mean parameters for distribution of underlying normal (variance is fixed at 1)
#' @export
#' @examples
#' data(Data.Test)
#' Data.Test.pairs <- Breaking(Data.Test, "full")
#' Estimation.Normal.GMM(Data.Test.pairs, 5)
def GMMGaussianRUM(data, m, n, itr=1):

    t0 = time.time() #get starting time

    muhat = np.ones((1, m), float)
    breaking = dataBreaking(data, m, False)

    for itr in range(1,itr + 1):
        try:
            Hinv = np.linalg.pinv(hessianPrPair(breaking, muhat, n))
        except np.linalg.linalg.LinAlgError:
            Hinv = 0.01 * np.identity(m)
        Grad = gradientPrPair(breaking, muhat, n)
        muhat = (muhat.transpose() - np.dot(Hinv, Grad)).transpose()
        muhat = muhat - muhat.min()

    t = time.time() - t0
    print("Time used:", t)

    return muhat
