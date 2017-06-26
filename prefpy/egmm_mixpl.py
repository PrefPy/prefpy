import numpy as np
import time
from plackettluce import *
from collections import defaultdict

def draw_kpl_partial(alphas, gammas, alts, head):
    k = len(alphas)
    cumalphas = np.cumsum(alphas)
    toss = np.random.rand()
    idx = 0
    for r in range(k):
        if toss <= cumalphas[r]:
            idx = r
            break
    gamma = gammas[idx]
    S = 1
    for ind in head:
        S -= gamma[int(ind)]
    l = len(alts)
    localgamma = []
    localalts = np.asarray(alts.copy())
    for i in range(l):
        localgamma.append(gamma[alts[i]])
    localgamma = np.asarray(localgamma)
    vote = []
    for i in range(l-1):
        localpr = []
        for g in localgamma:
            localpr.append(1/g)
        localpr = np.asarray(localpr)
        cumpr = np.cumsum(localpr/np.sum(localpr))
        lp = len(cumpr)
        toss2 = np.random.rand()
        for j in range(lp):
            if toss2 <= cumpr[j]:
                ind = j
                break
        vote.append(localalts[ind])
        localgamma = np.delete(localgamma, ind)
        localalts = np.delete(localalts, ind)
    vote.append(localalts[0])
    return vote
"""
To convert a weak order in the form of permutation ([1, 2, 2, 0] means a3>a0>a1=a2) to a strict order ([3,0,1,2] or [3,0,2,1])
ranking: the permutation to convert
alphas: mixing probabilities
gammas: parameters of all components
"""
def permconvert(ranking, alphas, gammas):
    flag = 0
    d = defaultdict(list)
    for i in range(len(ranking)):
        d[ranking[i]].append(i)
    vote = []
    for key in d.keys():
        if len(d[key]) == 1:
            vote=np.concatenate((vote,d[key]),axis = 0)
        else:
            flag = 1
            vote=np.concatenate((vote,draw_kpl_partial(alphas, gammas, d[key], vote)),axis = 0)
    return [ int(x) for x in vote ], flag

def rank2str(ranking):
    s = str(ranking[0])
    for alt in ranking[1:]:
        s += '-'+str(alt)
    return s

def Dictionarize(rankings, m):
    rankcnt = {}
    #print("1",rankings[1])
    for ranking in rankings:
        #print("2",ranking)
        flag = 0
        l = len(ranking)
        if len(set(ranking)) < l:
            print("Orders with duplicate alternatives are ignored!")
            continue
        for i in range(l):
            if ranking[i] >= m or ranking[i] < 0:
                flag = 1
        if flag == 1:
            print("Alternative index out of range! Ranking ignored!")
            continue
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
    ranking = ranking.split('-')
    l = len(ranking)
    s = 0
    for i in range(l):
        s += gamma[int(ranking[i])]
    p = 1
    for i in range(l-1):
        alt = int(ranking[i])
        p *= gamma[alt]/s
        s -= gamma[alt]
    return p


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


def egmm_mixpl(data, m, k = 2, itr = 20):
    if k == 1:
        itr = 2
    n = len(data)
    alphas = np.random.rand(k, 1)
    alphas /= np.sum(alphas)
    gammas = np.random.rand(k, m)
    for r in range(k):
        gammas[r] /= np.sum(gammas[r])
        if any(gammas[r]) <= 0.001:
            gammas[r] = renorm(gammas[r])
    rankings_strict = []
    rankings_weak = []
    perms_weak = []
    for j in range(len(data)):
        temp, idr = permconvert(data[j], alphas, gammas)
        if idr == 0:
            rankings_strict.append(temp)
        else:
            rankings_weak.append(temp)
            perms_weak.append(data[j])
    rankings = rankings_strict + rankings_weak
    for i in range(itr):
        if not perms_weak and i == 1:
            rankings_weak = []
            for j in range(len(perms_weak)):
                temp, idr = permconvert(perms_weak[j], alphas, gammas)
                rankings_weak.append(temp)
            rankings = rankings_strict + rankings_weak
        DataDict = Dictionarize(rankings, m)
        #print(rankings)
        #print(DataDict)
        if len(DataDict.keys()) == 0:
            print("No valid rankings!")
            return np.zeros((k, m+1))
        n1 = np.zeros((1, k), float)[0]
        breaking = np.zeros((k, m, m), float)
        #E Step
        for vote, freq in DataDict.items():
            #print(vote)
            ranking = vote.split('-')
            weights = np.zeros((1, k),float)[0]
            ss = 0
            for r in range(k):
                weights[r] = alphas[r]*prob_pl(gammas[r],vote)
                ss += weights[r]
            weights /= ss
            n1 += weights * freq
            for r in range(k):
                for i1 in range(0, m-1):
                    for i2 in range(i1+1, m):
                        #print(i1,i2)
                        breaking[r][int(ranking[i1]), int(ranking[i2])] += freq*weights[r]#/(m-i1)
        #M Step
        alphas = n1/np.sum(n1)
        #print(alphas)
        for r in range(k):
            gammas[r] = aggpl(breaking[r])
            for i in range(m):
                if gammas[r][i] <= 0.001:
                    gammas[r] = renorm(gammas[r])
    #print("alphas = ", alphas)
    #print("gammas = ", gammas)
    rslt = np.concatenate((np.array([alphas]).T,gammas), axis=1)
    return rslt
