import numpy as np
from scipy import stats
import time
import heapq
import random
from itertools import product
import cPickle as pickle

class GMMError(Exception):
    def __init__(self, issue, reproduce):
        self.issue = issue
        self.reproduce = reproduce

    

def interpret_rank_data(filename):
    """Takes name for text file with data in form NAMES\n TYPES\n DATA with D dimensions and L labels.
             Returns data in the form [(x_1..x_D)],[(l_1..l_L)]"""
    X = []
    L = []
    datafile = open(filename, 'r')
    header = datafile.readline().split('\t') # Get first line, split
    header[-1] = header[-1].rstrip() # Get rid of trailing newline
    ndim = 0
    nlab = 0
    for elt in header:
        if elt[0] == 'L':
            nlab += 1
        if elt[0] == 'A':
            ndim += 1
    datafile.readline() # Skip the next line, it's really not necessary
    raw_data = datafile.readlines()
    for line in raw_data:
        thisX = []
        thisL = []
        if line[0] == '\r': # TODO: Make this more graceful
            continue
        line = line.rstrip().split('\t')
        for i in range(ndim):
            thisX.append(float(line[i]))
        for i in range(nlab):
            thisL.append(int(line[i+ndim]))
        X.append(thisX)
        L.append(thisL)
    datafile.close()
    return np.array(X), L

class IB_PLGMM:
    def __init__(self, breaking='full', K=0, weighting=lambda x: 1, useborda=False, k=3):
        """ Initialize, taking breaking as option """
        self.breaking = breaking
        self.K = K
        self.k = k
        self.weighting = weighting
        self.useborda = useborda

    def train(self, X, L):
        """ "Train" the model by loading the data and setting k, m"""
        # This pretty much constitutes training
        self.X = X
        self.L = L
        self.dataset = zip(self.X, self.L)
        #self.k = int(np.sqrt(X.shape[0]))
        self.m = len(L[0])

    def _row_in(self, point):
        for i in self.dataset:
            if all(np.abs(i[0] - point) < 1e-9):
                print "Found eerily similar point:", i, point
                return True
        return False

    def _get_k_nearest(self, k, point):
        """ Get k nearest neighbors of a point O(nlogk + klogk)"""
        #self._row_in(point)
        dist = lambda x: np.linalg.norm(point - x[0]) # Assuming a zip will be given
        init_heap = [(dist(i), i) for i in self.dataset[:k]]
        for instance in self.dataset:
            cur_dist = dist(instance)
            if cur_dist < init_heap[0][0]:
                try:
                    heapq.heappop(init_heap)
                except ValueError:
                    raise GMMError('heappop',tuple(init_heap))
                try:
                    heapq.heappush(init_heap, (cur_dist, instance))
                except ValueError:
                    raise GMMError('heappush', (init_heap, (cur_dist, instance)))
        return [i[1] for i in sorted(init_heap, key=lambda x: x[0])]

    def _full(self, k):
        """ Full breaking """
        # doesn't do anything with k
        G = np.ones((self.m, self.m))
        np.fill_diagonal(G, 0)
        return G

    def _top(self, k):
        """ Top k breaking """
        if k > self.m:
            raise ValueError
        G = np.ones((self.m, self.m))
        np.fill_diagonal(G, 0)
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue
                if i > k and j > k:
                    G[i][j] = 0
        return G

    def _bot(self, k):
        """ Bottom k breaking """
        if k < 2:
            raise ValueError
        G = np.ones((self.m, self.m))
        np.fill_diagonal(G, 0)
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue
                if i <= k and j <= k:
                    G[i][j] = 0
        return G

    def _adj(self, k):
        """ Adjacent breaking """
        # doesn't do anything with k
        G = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if (i == j+1) or (j == i+1):
                    G[i][j] = 1
        return G

    def _pos(self, k):
        """ Position k breaking """
        if k < 2:
            raise ValueError
        G = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in range(self.m):
                if i == j:
                    continue
                if i < k or j < k:
                    continue
                if i == k or j == k:
                    G[i][j] = 1
        return G

    def _get_GMM(self, neighbors, point):
        """ Given all of the neighbors in the form [(X, L)], get the GMM estimate of the parameters
                        Can take breakings as defined in Lirong's paper """
        breakings = {
                        'full':self._full,
                        'top':self._top,
                        'botk':self._bot,
                        'adj':self._adj,
                        'posk':self._pos
        }
        # Get a breaking
        breaking_adjmat = breakings[self.breaking](self.K)
        P = np.zeros((self.m, self.m))
        # Compute list of distances
        dist = lambda x: np.linalg.norm(point - x)
        distance_list = np.array([dist(i[0]) for i in neighbors])
        max_dist = max(distance_list)
        # Normalize it to [0,1]
        distance_list /= max_dist

        for ind, neighbor in enumerate(neighbors):
            # First get ranking matrix X (rm for scope reasons)
            rm = np.zeros((self.m, self.m))
            for i in range(self.m):
                for j in range(self.m):
                    alt_i_pos = neighbor[1].index(i+1) # silliness with 1-indexing of alternatives
                    alt_j_pos = neighbor[1].index(j+1)
                    if alt_i_pos < alt_j_pos: # i.e., alt_i > alt_j in the ranking since indices are backwards
                        rm[i][j] = 1
            rm *= breaking_adjmat # Essentially AND it with the adjacency matrix representing the breaking
            # Construct local P from rm
            localP = rm[:]
            # Essentially compute the values in the diagonal, and fill it
            for i in range(self.m):
                localP[i][i] = -1*(np.sum(rm.T[i][:i]) + np.sum(rm.T[i][i+1:])) # quick and dirty way to do sum w/o i
            # Before we can weight it by distance, we need to get
            # Now weight it. Works as 1/weighting
            localP *= self.weighting(distance_list[ind])/float(len(neighbors))
            P += localP
        # Now that we have P_G(D), we can solve the system P_G(D)*gamma = 0 with an svd of P_G(D)
        epsilon = 1e-7 # Close enough to 0 for 32bit precision
        userank = True
        try:
            assert(np.linalg.matrix_rank(P) == self.m-1)
        except AssertionError:
            print np.linalg.matrix_rank(P), self.m, P
            userank = False
        try:
            assert(all(np.sum(P, axis=0) <= epsilon))
        except AssertionError:
            print np.sum(P, axis=0)
            raise GMMError('colstoch',(P, neighbors))
        U, S, V = np.linalg.svd(P)
        # Found on stackoverflow
        if not userank:
            gamma = np.abs(np.compress(S <= epsilon, V, axis=0).T)
            try:
                assert(np.rank(gamma) == 1)
            except AssertionError:
                print np.rank(gamma), gamma
                raise GMMError('ambiggamma',(gamma, P, neighbors))
        # John's method, since rank is supposedly m-1
        else:
            gamma = np.abs(V[-1])
        try:
            assert(all(np.dot(P, gamma) < epsilon))
        except AssertionError:
            print np.dot(P, gamma)
            raise GMMError('notnullspace',(gamma, P, neighbors, epsilon))

        return gamma

    def _vote_borda(self, neighbors, point):
        # Compute list of distances
        #print neighbors[0][0], point
        dist = lambda x: np.linalg.norm(point - x)
        distance_list = np.array([dist(i[0]) for i in neighbors])
        max_dist = max(distance_list)
        # Normalize it to [0,1]
        distance_list /= max_dist
        borda_index = [0 for i in range(self.m)]
        for ind, neighbor in enumerate(neighbors):
            for i in range(self.m):
                position = neighbor[1].index(i+1)
                borda_index[i] += (self.m - position)*self.weighting(distance_list[ind])
        #print borda_index
        return borda_index

    def rank(self, point, useborda=False):
        """ Figure out the PL parameters of a point using the GMM method """
        # First get neighbors (including self)
        ticret = time.time()
        neighbors = self._get_k_nearest(self.k, point)
        tocret = time.time() - ticret
        #print "Took %f seconds to get the neighbors" % tocret
        # Then pass neighbors (including self) into the GMM function, specifying a breaking, to get a list of parameters
        gamma = None
        if not useborda:
            ticgmm = time.time()
            gamma = self._get_GMM(neighbors, point)
            tocgmm = time.time() - ticgmm
            #print "Took %f seconds to get the parameters" % tocgmm
            # Then sort the returned gammas (with each gamma's index corresponding to their label) to get the MAP estimate
            try:
                sorted_labels = [i+1 for i in sorted(range(len(gamma)), key=lambda x: -gamma[x])]
            except ValueError:
                raise GMMError('gammamult',tuple(gamma))
        else:
            borda_score = self._vote_borda(neighbors, point)
            sorted_labels = [i[0]+1 for i in sorted(enumerate(borda_score), key=lambda x: -x[1])]
        return sorted_labels, gamma

    def correlation(self, ranking1, ranking2):
        """ tau-b """
        return stats.kendalltau(ranking1, ranking2)

    def correlation2(self, ranking1, ranking2):
        """ Standard tau"""
        c = 0
        d = 0
        for ind, i in enumerate(ranking1):
            for ind2, j in enumerate(ranking1):
                if ind == ind2:
                    continue
                if ind > ind2 and ranking2.index(i) > ranking2.index(j):
                    c += 1
                if ind < ind2 and ranking2.index(i) < ranking2.index(j):
                    c += 1
                if ind > ind2 and ranking2.index(i) < ranking2.index(j):
                    d += 1
                if ind < ind2 and ranking2.index(i) > ranking2.index(j):
                    d += 1
        return float(c - d)/(self.m*(self.m-1)*0.5), 0
           




    def score(self, realX=None, realL=None, useborda=False):
        """ Compute average kendall tau distance of training data by going through self.X, self.L """
        if realX == None:
            realX = self.X
        if realL == None:
            realL = self.L
        # Essentially go through each of X, L and rank
        # Assume that realX and realL are in the same order, i.e. X[0]'s rank is L[0]
        ktavg = 0
        tottic = time.time()
        for ind, instance in enumerate(realX):
            tic = time.time()
            #print "Ranking instance %d" % ind
            try:
                predicted, gamma = self.rank(instance, useborda=useborda)
            except GMMError as gme:
                raise gme
            toc = time.time() - tic
            #print "Took %f seconds" % toc
            #ktd, pv = self.correlation(predicted, realL[ind])
            ktd, pv = self.correlation2(predicted, realL[ind]) # Using the standard tau instead of tau-b
            #print "Got Kendall Tau correlation of %f" % ktd
            ktavg += ktd
        tottoc = time.time() - tottic
        ktavg /= len(realX)
        #print "Average Kendall Tau: %f" % ktavg
        #print "Took %f seconds (%f seconds on average)" % (tottoc, tottoc/len(realX))
        return ktavg

def simple_cross_val(dataset, k, gmm):
    # randomize dataset
    random.shuffle(dataset)
    n = float(len(dataset))
    # split the dataset into k equal parts
    ktau_scores = []
    for i in range(k):
        # Fancy slicing
        test = dataset[int(i*n/k):int((i+1)*n/k)]
        train = dataset[0:int(i*n/k)] + dataset[int((i+1)*n/k):-1]
        # Madness with numpy arrays and zip
        trainX, trainL = zip(*train)
        trainX = np.array(trainX)
        testX, testL = zip(*test)
        testX = np.array(testX)
        gmm.train(trainX, trainL)
        try:
            ktscore = gmm.score(realX=testX, realL=testL, useborda=gmm.useborda)
        except GMMError as gme:
            gme.reproduce = gme.reproduce + (train, test)
            raise gme
        ktau_scores.append(ktscore)
        # Have fun gc...
    return np.array(ktau_scores)

def run(folds=10, iterations=5, dataset='LabelRankingData/wine_dense.txt', weighting=lambda x: 1, k=3, useborda=False):
    X, L = interpret_rank_data(dataset)
    useful_dataset = zip(X,L)
    simple_gmm = IB_PLGMM(weighting=weighting, k=k, useborda=useborda)

    simple_gmm_scores = []
    for i in range(iterations):
        try:
            simple_gmm_scores.append(np.average(simple_cross_val(useful_dataset, folds, simple_gmm)))
        except GMMError as gme:
            raise gme

    return np.array(simple_gmm_scores)

if __name__ == "__main__":
    function_dict = {
                     'identity': lambda x: 1,
                     'inverse': lambda x: 1 - x + 0.001, #Otherwise k=1 will never work because the normalized distance will always be 1, and thus the neighbor will be weighted at 0
                     'invexp': lambda x: np.exp(-x),
                     'neglog': lambda x: -np.log(1e-9+x)
                     }

    test_datasets = [#'analcatdata-authorship_dense.txt',
                     'bodyfat_dense.txt',
                     'glass_dense.txt',
                     'housing_dense.txt',
                     'iris_dense.txt',
                     #'stock_dense.txt',
                     #'vehicle_dense.txt',
                     'vowel_dense.txt',
                     'wine_dense.txt',
                     'wisconsin_dense.txt'
                     ]
    test_datasets = ['LabelRankingData/' + i for i in test_datasets]
    max_k = 10
    test_ks = [i for i in range(2,max_k) if i % 2 != 0]
    bordas = [True, False]

    weighting_funcs = ['identity', 'inverse', 'invexp', 'neglog']
    grid = product(test_datasets, test_ks, weighting_funcs, bordas)
    result_dict = {d:{k:{w:{b:-2 for b in bordas} for w in weighting_funcs} for k in test_ks} for d in test_datasets}
    for paramset in grid:
        print "LOG: Dataset %s\t k: %d\t weighting: %s\t borda: %r" % paramset
        tic = time.time()
        try:
            score = run(dataset=paramset[0], k=paramset[1], weighting=function_dict[paramset[2]], useborda=paramset[3])
            toc = time.time() - tic
            print "LOG: Took %f seconds: scores %r" % (toc, score)
            result_dict[paramset[0]][paramset[1]][paramset[2]][paramset[3]] = score
        except GMMError as gme:
            print "Encountered error with", paramset, "noted and continuing"
            result_dict[paramset[0]][paramset[1]][paramset[2]][paramset[3]] = (gme.issue, gme.reproduce)
    dumpfile = open('resultsmorestdtau.dat','w')
    pickle.dump(result_dict, dumpfile)
    dumpfile.close()
