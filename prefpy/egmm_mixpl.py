import numpy as np
import time

def rank2str(ranking):
    s = str(int(ranking[0]))
    for alt in ranking[1:]:
        s += '-'+str(int(alt))
    return s

def Dictionarize(rankings):
    rankcnt = {}
    #print("1",rankings[1])
    for ranking in rankings:
        key = rank2str(ranking)
        if key in rankcnt:
            rankcnt[key] += 1
        else:
            rankcnt[key] = 1
    return rankcnt

def renorm(gamma):
    l = len(gamma)
    ind = np.argmax(gamma)
    for i in range(l):
        if gamma[i] <= 0.001:
            d = gamma[i] - 0.001
            gamma[i] = 0.001
            gamma[ind] += d
    return gamma

def prob_pl(gamma, ranking):
    m = len(gamma)
    s = np.sum(gamma)
    p = 1
    for i in range(m-1):
        alt = int(ranking[i])
        p *= gamma[alt]/s
        s -= gamma[alt]
    return p

def pr_mixpl(theta, ranking):
    k = len(theta)
    m = len(theta[0]) - 1
    pr = 0
    for r in range(k):
        pr += theta[r][0]*prob_pl(theta[r][1:], ranking)
    return pr

def nllpl(theta, data):
    nn = len(data)
    ll = 0
    for j in range(nn):
        ll += data[j][0]*np.log(pr_mixpl(theta, data[j][1:]-1))
    nll = -ll
    return nll

def aggpl(breaking):
    m = len(breaking)
    for ind in range(0, m):
        breaking[ind][ind] = -(np.sum(breaking.T[ind]))
    U, S, V = np.linalg.svd(breaking)
    gamma = np.abs(V[-1])
    gamma /= np.sum(gamma)
    return gamma

'''
	Title: EGMM Function

	Function: Algorithm

	Input:
		data - the rankings
				numpy array of rankings, see generate GenerateRUMData for data generation.
		n 	 - the number of agents (rankings)
				integer
		m	 - the number of alternatives (candidates)
				integer
		k	 - the number of components
				integer
				must be greater than or equal to 1
				default is 2
		itr	 - the number of iterations
				integer
				default is 10

	Output:

'''


def egmm_mixpl(data, k = 2, itr = 20):
    n = len(data)
    m = len(data[0])
    alphas = np.random.rand(k)
    alphas /= np.sum(alphas)
    gammas = np.random.rand(k, m)
    for r in range(k):
        gammas[r] /= np.sum(gammas[r])
        if any(gammas[r]) <= 0.001:
            gammas[r] = renorm(gammas[r])
    te = 0
    tm = 0
    DataDict = Dictionarize(data)
    for i in range(itr):
        n1 = np.zeros((1, k), float)[0]
        breaking = np.zeros((k, m, m), float)
        #E Step
        t0 = time.perf_counter()
        for vote, freq in DataDict.items():
            ranking = vote.split('-')
            weights = np.zeros((1, k),float)[0]
            ss = 0
            for r in range(k):
                weights[r] = alphas[r]*prob_pl(gammas[r],ranking)
                ss += weights[r]
            weights /= ss
            n1 += weights * freq
            for r in range(k):
                for i1 in range(0, m-1):
                    for i2 in range(i1+1, m):
                        breaking[r][int(ranking[i1]), int(ranking[i2])] += freq*weights[r]#/(m-i1)
        t1 = time.perf_counter()
        #M Step
        alphas = n1/np.sum(n1)
        for r in range(k):
            gammas[r] = aggpl(breaking[r])
            for i in range(m):
                if gammas[r][i] <= 0.001:
                    gammas[r] = renorm(gammas[r])
        t2 = time.perf_counter()
        #print("alphas:", alphas)
        #print("gammas:", gammas)
        te += t1 - t0
        tm += t2 - t1
    rslt = np.hstack((alphas, np.reshape(gammas, (1, k*m))[0]))
    return rslt, te, tm
