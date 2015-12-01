# Implementation of the algorithm from
# Random Utility Theory for Social Choice
# by Azari, Parkes, & Xia
# This file implements the unknown variance,
# unknown mean case for Normal distribution

import numpy as np
from scipy import stats


def mean_squared_error(mean, estimator):
    """
    Description:
        Calculates the Mean Squared Error (MSE) of
        an estimation
    Parameters:
        mean:      actual value
        estimator: estimated value of the mean
    """
    return np.average((estimator - mean) ** 2)

def estimate_theta(n, m, pi, N, epsilon, max_iters, true_theta):
    """
    Description:
        Monte-Carlo Expectation-Maximization Algorithm:
        Returns an estimates of the ground-truth parameters,
        theta for the given data
    Parameters:
        n:          number of agents
        m:          number of alternatives
        pi:         agents' orderings over alternatives (n by m array)
        N:          number of Gibbs samples at each iteration
        epsilon:    convergence condition
        max_iters:  maximum number of iterations of MC-EM algorithm
        true_theta: ground truth parameters (simply in order to calculate MSE)
    """
    # generate initial value for theta, i.e. theta_t0:

    # generate mu (means) randomly centered at 1.0
    mu_t = np.random.normal(loc=0, scale=1, size=m)

    # generate sigmas randomly with a truncated norm
    # centered at 1.0 with standard deviation of 1,
    # and truncated at 0 (so only positive values)
    sigma_t = stats.truncnorm.rvs(a=np.array([-1]),
                                  b=np.array([stats.norm.b]),
                                  loc=1, scale=1, size=m)

    # combine into theta_t with dimension (m, 2)
    theta_t = np.vstack(([mu_t], [sigma_t])).T
    
    for f in range(max_iters):
        S_t = estep(n, m, pi, N, theta_t)
        theta_t1 = mstep(n, m, S_t)
        mse = mean_squared_error(true_theta, theta_t1)
        print("Iter {}:\tMSE = {}".format(f+1, mse))
        print("Theta_t+1:", theta_t1)

        if np.all(np.absolute(theta_t1 - theta_t) < epsilon):
            break # convergence reached before max_iters

        theta_t = theta_t1
        
    return theta_t1

def approx_S(n, m, pi, N, theta_t, x_0):
    """
    Description:
        Returns an estimate of the n by m array S by performing
        Gibbs sampling agents' latent utility of all alternatives
    Parameters:
        n:       number of agents
        m:       number of alternatives
        pi:      agents' orderings over alternatives (n by m array)
        N:       number of Gibbs samples
        theta_t: current estimate of the parameters
        x_0:     initial value for the utilities for Gibbs sampling
    """
    x_N = []
    x_N.append(x_0)
    for k in range(N):
        x_k = np.zeros((n, m)) + np.nan
        for j in range(m):
            theta_j = theta_t[j]
            for i in range(n):
                # hacky way to find index where value is j:
                alt_ind = np.where(pi[i] == j)[0][0]
                
                next_best = pi[i][alt_ind - 1] if alt_ind > 0 else None
                next_worst = pi[i][alt_ind + 1] if alt_ind < (len(pi[i]) - 1) else None
                
                if next_best == None:
                    rval = stats.norm.b # +inf
                elif next_best < j:
                    rval = x_k[i][next_best] - theta_j[0]
                else: # next_best > j
                    rval = x_N[k][i][next_best] - theta_j[0]

                if next_worst == None:
                    lval = stats.norm.a # -inf
                elif next_worst < j:
                    lval = x_k[i][next_worst] - theta_j[0]
                else: # next_worst > j
                    lval = x_N[k][i][next_worst] - theta_j[0]

                distr =  stats.truncnorm(a=np.array([lval]), b=np.array([rval]), loc=theta_j[0], scale=theta_j[1])
                sample = distr.rvs() # sample the truncated Normal distribution
                x_k[i][j] = sample

        x_N.append(x_k)


    # can alter code below to discard samples
    T_x_avg = np.sum(x_N, axis=0) / N
    T_x2_avg = np.sum(np.square(x_N), axis=0) / N
    return np.vstack(([T_x_avg], [T_x2_avg])) # combine to form S with dimension (2, n, m)
                    

def estep(n, m, pi, N, theta_t):
    """
    Description:
        The E-step which approximately computes S_t1 using a
        Gibbs sampler given the current estimate of the
        parameters theta_t and conditioned on the data pi so
        that the M-step can compute theta_t1
    Parameters:
        n:       number of agents
        m:       number of alternatives
        pi:      agents' orderings over alternatives (n by m array)
        N:       maximum number of iterations
        theta_t: estimation of the parameters from the previous M-step
    """

    # generate initial "sample" of latent utilities x
    x_0 = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            # hacky way to find index where value is j:
            pi_i_j_index = np.where(pi[i] == j)[0][0]
            x_0[i][j] = (m - pi_i_j_index) / m # normalize the utilities by dividing by m

    S_t = approx_S(n, m, pi, N, theta_t, x_0)
    # Q = ? # not needed for fixed variance case, can just use S_t
    return S_t

def mstep(n, m, S_t):
    """
    Description:
        The M-step which computes theta_t1 by maximizing having beeen
        given an estimation of S_t from the E-step
    Parameters:
        n:   number of agents
        m:   number of alternatives
        S_t: approximation of S_t from preceeding E-step
    """
    mu_t1 = np.average(S_t[0], axis=0)

    # take sqrt because my closed-form solutions were for
    # variance (usual paramater of Normal), but cheating a
    # little in this code by mainly using standard deviation
    # instead
    sigma_t1 = np.sqrt(-(mu_t1**2) + np.average(S_t[1], axis=0))
    theta_t1 = np.vstack(([mu_t1], [sigma_t1])).T
    return theta_t1

def test1():
    """
    50 agents
    3 alternatives
    250 iterations in round-robin Gibbs sampler
    500 maximum iterations of MC-EM
    1*10**-12 epsilon convergence condition
    """
    n = 50
    m = 3
    zeta = np.random.normal(loc=0, scale=0.3, size=(n, m)) # agents' subjective noise
    actual_mu = np.random.normal(loc=0, scale=1, size=m)
    actual_sigma = stats.truncnorm.rvs(a=np.array([-1]),
                                  b=np.array([stats.norm.b]),
                                  loc=1, scale=1, size=m)
    actual_theta = np.vstack(([actual_mu], [actual_sigma])).T
    X = actual_mu + zeta
    pi = []
    for i in range(n):
        pi.append(sorted([j for j in range(m)], key=lambda x: -X[i][x]))

    pi = np.array(pi)
    N = 250
    epsilon = 1e-12
    max_iters = 500
    print(actual_theta)
    print(X)
    print(pi)
    theta = estimate_theta(n, m, pi, N, epsilon, max_iters, actual_theta)
    print(theta)

def test2():
    pass
    

def main():
    """
    Driver function for the computation
    of the MC-EM algorithm given the filename
    in which the data resides and various parameters
    for the fine-tuning of the algorithm's accuracy
    """
    test1()
    #test2()


if __name__ == "__main__":
    main()
