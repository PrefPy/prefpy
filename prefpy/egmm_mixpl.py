import numpy as np				  # used for matrices and vectors
from .plackettluce import *
from collections import defaultdict

"""
To draw a partial order given the parameter.
alphas: mixing probabilities.
gammas: k-by-m matrix. Each row is the parameter of a component.
alts: the subset of alternatives that we want to sample a partial order.
output: a partial ranking
"""

def draw_kpl_partial(alphas, gammas, alts):
    k = len(alphas)
    cumalphas = np.cumsum(alphas)
    toss = np.random.rand()
    idx = 0
    for r in range(k):
        if toss <= cumalphas[r]:
            idx = r
            break
    gamma = gammas[idx]
    l = len(alts)
    localgamma = []
    localalts = np.asarray(alts.copy())
    for i in range(l):
        localgamma.append(gamma[alts[i]])
    localgamma = np.asarray(localgamma)
    vote = []
    for i in range(l-1):
        cumgamma = np.cumsum(localgamma/np.sum(localgamma))
        lp = len(cumgamma)
        toss2 = np.random.rand()
        for j in range(lp):
            if toss2 <= cumgamma[j]:
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
            vote=np.concatenate((vote,draw_kpl_partial(alphas, gammas, d[key])),axis = 0)
    return [ int(x) for x in vote ], flag


"""
	Convert a ranking to a string so that it can be used as keys to dictionaries.
"""

def rank2str(ranking):
	s = ""
	for alt in ranking:
		s += str(alt)
	return s

"""
	Given a dataset, this function returns a dictionary with rankings as keys and frequencies of each ranking as values
"""

def Dictionarize(rankings, m):
	rankcnt = {}
	for ranking in rankings:
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
	#print(gamma)
	m = len(gamma)
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

def freqBreaking(weights, m, k):
	breaking = np.zeros((k, m, m), float)
	for r in range(k):
		for vote, freq in weights[r].items():
			l = len(vote)
			for i1 in range(0, l-1):
				for i2 in range(i1+1, l):
					breaking[r][int(vote[i1]), int(vote[i2])] += freq
	return breaking

def aggpl(breaking):
	m = len(breaking)
	for ind in range(0, m):
		breaking[ind][ind] = -(np.sum(breaking.T[ind]))
	#print(breaking)
	U, S, V = np.linalg.svd(breaking)
	gamma = np.abs(V[-1])
	gamma /= np.sum(gamma)
	return gamma

'''
	Title: EGMM Function

	Function: Algorithm

	Input:
		data - permutaions with the each ranking (eg. a1>a0>a2=a3) represented as list of postions ([1, 0, 2, 2])
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


def egmm_mixpl(data, m, k = 2, itr = 10):
    if k == 1:
        itr = 2
    n = len(data)
    alphas = np.random.rand(k,1)
    alphas /= np.sum(alphas)
    gammas = np.random.rand(k, m)
    for r in range(k):
        gammas[r] /= np.sum(gammas[r])
        if any(gammas[r]) <= 0.001:
            gammas[r] = renorm(gammas[r])
    breaking = np.zeros((k, m, m), float)
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
        #Resample weak orders
        if not perms_weak and i == 1:
            rankings_weak = []
            for j in range(len(perms_weak)):
                temp, idr = permconvert(perms_weak[j], alphas, gammas)
                rankings_weak.append(temp)
            rankings = rankings_strict + rankings_weak
        DataDict = Dictionarize(rankings, m)
        if len(DataDict.keys()) == 0:
            print("No valid rankings found in the data!")
            return np.zeros((k, m+1))
        weights = []
        for r in range(k):
            weights.append(DataDict.copy()) # The length of each dictionary is the same as the dataset dictionary
        n1 = np.zeros((1, k), float)[0]
        for vote, freq in DataDict.items():
            ss = 0
            for r in range(k):
                weights[r][vote] = alphas[r][0]*prob_pl(gammas[r],vote)
                ss += weights[r][vote]
            for r in range(k):
                weights[r][vote] *= freq/ss
        breaking = freqBreaking(weights, m, k)
        total = 0
        for r in range(k):
            for key, value in weights[r].items():
                n1[r] += value
            total += n1[r]
        flag = 0
        for r in range(k):
            alphas[r] = n1[r]/total
            gammas[r] = aggpl(breaking[r])
            for i in range(m):
                if gammas[r][i] <= 0.001:
                    flag = 1
                    gammas[r] = renorm(gammas[r])
        rslt = np.concatenate((alphas,gammas), axis=1)
    if flag == 1:
        print("Data is ill-conditioned for k-PL, result may be inaccurate. Try again or reduce the number of components!")
    return rslt

if __name__ == "__main__":
    param = np.array([[0.1,0.2,0.3,0.5],[0.3,0.4,0.2,0.4],[0.6,0.3,0.5,0.2]])
    alphas = [0.1,0.3,0.6]
    gammas = [[0.2,0.3,0.5],[0.4,0.2,0.4],[0.3,0.5,0.2]]
    ranking = [1,1,0]
    print(ranking)
    print(permconvert(ranking,alphas,gammas))
