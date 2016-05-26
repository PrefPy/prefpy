# Filename: gmm_mixpl.py
# Author: Peter Piech
# Date: 1/14/2016
# Description: Implementation of Algorithm 1 from
#              "Learning Mixtures of Plackett-Luce models"
#              by Zhao, Piech, & Xia (2016) which is a
#              Generalized Method of Moments (GMM) Algorithm
#              for rank data using the MatLab for Python
#              runtime with fallback for SciPy.

import numpy as np
import scipy
import os
import time
import importlib
from collections import namedtuple
import functools
from . import aggregate
from . import plackettluce as pl
from . import gmm_mixpl_moments as mixpl_moments
from . import gmm_mixpl_objectives as mixpl_objs
from . import stats

_matlab_support = True
try:
    matlab = importlib.import_module("matlab")
    matlab.engine = importlib.import_module("matlab.engine")
except ImportError:
    _matlab_support = False

GMMMixPLFunctions = namedtuple("GMMMixPLFunctions", "calcMoments mixPLobjective")

def calcMomentsMatlabEmpirical(params):
    """Top 3 alternatives 20 empirical moment conditions"""
    alpha = params[0]
    a = params[1:5]
    b = params[5:]
    p1 = alpha*a+(1-alpha)*b
    p21 = alpha*a[0]*a[1:]/(1-a[0])+(1-alpha)*b[0]*b[1:]/(1-b[0])
    p22 = alpha*a[1]*np.hstack((a[0],a[2:]))/(1-a[1])+(1-alpha)*b[1]*np.hstack((b[0],b[2:]))/(1-b[1])
    p23 = alpha*a[2]*np.hstack((a[:2],a[3]))/(1-a[2])+(1-alpha)*b[2]*np.hstack((b[:2],b[3]))/(1-b[2])
    p24 = alpha*a[3]*a[:3]/(1-a[3])+(1-alpha)*b[3]*b[:3]/(1-b[3])
    p3 = np.array([
        alpha*a[0]*a[2]*a[3]/(1-a[2])/(a[0]+a[1])+(1-alpha)*b[0]*b[2]*b[3]/(1-b[2])/(b[0]+b[1]),
        alpha*a[0]*a[1]*a[3]/(1-a[3])/(a[1]+a[2])+(1-alpha)*b[0]*b[1]*b[3]/(1-b[3])/(b[1]+b[2]),
        alpha*a[0]*a[1]*a[2]/(1-a[0])/(a[3]+a[2])+(1-alpha)*b[0]*b[1]*b[2]/(1-b[0])/(b[3]+b[2]),
        alpha*a[2]*a[1]*a[3]/(1-a[1])/(a[0]+a[3])+(1-alpha)*b[2]*b[1]*b[3]/(1-b[1])/(b[0]+b[3])
        ])
    return np.concatenate((p1,p21,p22,p23,p24,p3))

def calcMomentsMatlabEmpirical_reduced(params):
    """Top 3 alternatives 16 empirical moment conditions"""
    alpha = params[0]
    a = params[1:5]
    b = params[5:]
    p1 = alpha*a+(1-alpha)*b
    p21 = alpha*a[0]*a[2:]/(1-a[0])+(1-alpha)*b[0]*b[2:]/(1-b[0])
    p22 = alpha*a[1]*np.hstack((a[0],a[3]))/(1-a[1])+(1-alpha)*b[1]*np.hstack((b[0],b[3]))/(1-b[1])
    p23 = alpha*a[2]*a[:2]/(1-a[2])+(1-alpha)*b[2]*b[:2]/(1-b[2])
    p24 = alpha*a[3]*a[1:3]/(1-a[3])+(1-alpha)*b[3]*b[1:3]/(1-b[3])
    p3 = np.array([
        alpha*a[0]*a[2]*a[3]/(1-a[2])/(a[0]+a[1])+(1-alpha)*b[0]*b[2]*b[3]/(1-b[2])/(b[0]+b[1]),
        alpha*a[0]*a[1]*a[3]/(1-a[3])/(a[1]+a[2])+(1-alpha)*b[0]*b[1]*b[3]/(1-b[3])/(b[1]+b[2]),
        alpha*a[0]*a[1]*a[2]/(1-a[0])/(a[3]+a[2])+(1-alpha)*b[0]*b[1]*b[2]/(1-b[0])/(b[3]+b[2]),
        alpha*a[2]*a[1]*a[3]/(1-a[1])/(a[0]+a[3])+(1-alpha)*b[2]*b[1]*b[3]/(1-b[1])/(b[0]+b[3])
        ])
    return np.concatenate((p1,p21,p22,p23,p24,p3))

class GMMMixPLResult:
    def __init__(self, num_alts, num_votes, num_mix, true_params, cond, opto, soln_params, momnts_runtime, opto_runtime, overall_runtime):
        self.num_alts = num_alts
        self.num_votes = num_votes
        self.num_mix = num_mix
        self.true_params = true_params
        self.cond = cond
        self.opto = opto
        self.soln_params = soln_params
        self.momnts_runtime = momnts_runtime
        self.opto_runtime = opto_runtime
        self.runtime = overall_runtime


class GMMMixPLAggregator(aggregate.RankAggregator):
    """
    Generalized Method-of-Moments algorithm
    for Mixtures of Plackett-Luce models
    """

    mixPLalgorithms = {"top2_min":         GMMMixPLFunctions(mixpl_moments.top2_reduced, mixpl_objs.top2_reduced),               # 12 moments
                       "top2_full":        GMMMixPLFunctions(mixpl_moments.top2_full, mixpl_objs.top2_full),                     # 16 moments
                       "top2_min_uncons":  GMMMixPLFunctions(mixpl_moments.top2_reduced, mixpl_objs.top2_reduced_unconstrained), # 12 moments unconstrained
                       "top2_full_uncons": GMMMixPLFunctions(mixpl_moments.top2_full, mixpl_objs.top2_full_unconstrained),       # 16 moments unconstrained
                       "top3_min":         GMMMixPLFunctions(mixpl_moments.top3_reduced, mixpl_objs.top3_reduced),               # 16 moments
                       "top3_full":        GMMMixPLFunctions(mixpl_moments.top3_full, mixpl_objs.top3_full),                     # 20 moments
                       "top3_min_uncons":  GMMMixPLFunctions(mixpl_moments.top3_reduced, mixpl_objs.top3_reduced_unconstrained), # 16 moments unconstrained
                       "top3_full_uncons": GMMMixPLFunctions(mixpl_moments.top3_full, mixpl_objs.top3_full_unconstrained),       # 20 moments unconstrained
                      }

    def __init__(self, alts_list, use_matlab=False):
        super().__init__(alts_list)
        self.bounds_pairs = [(0.0, 1.0) for i in range(2*self.m + 1)]
        self.cons = ({"type": "eq",
                      "fun": lambda x: 1 - np.sum(x[1:self.m+1])},
                     {"type": "eq",
                      "fun": lambda x: 1 - np.sum(x[self.m+1:])}
                    )
        self.matlabEng = None
        if _matlab_support and use_matlab:
            self.matlabEng = matlab.engine.start_matlab()
            self.lb = matlab.double(np.zeros((9,1)).tolist())
            self.ub = matlab.double(np.ones((9,1)).tolist())
            self.A = matlab.double([])
            self.b = matlab.double([])
            self.Aeq = matlab.double(np.array([[0, 1, 1, 1, 1, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0, 1, 1, 1, 1]]).tolist())
            self.beq = matlab.double(np.ones((1,2)).tolist())
            self.Aeq_uncons = matlab.double([])
            self.beq_uncons = matlab.double([])

        # set matlab directory to the folder containing this module and thus also "optimize.m"
        self.matlabEng.cd(os.path.dirname(__file__), nargout=0)

    def aggregate(self, rankings, algorithm, epsilon, max_iters, approx_step, opto="scipy", true_params=None):
        """
        Description:
            Takes in a set of rankings and an algorithm to apply
            and computes the mixing proportions and parameters for
            a mixture of Plackett-Luce models.
        Parameters:
            rankings:    set of rankings over the set of alternatives
            algorithm:   string abbreviation of the moment conditions to use (scipy only)
            epsilon:     convergence condition threshold value (scipy only)
            max_iters:   maximum number of iterations for the optimization (scipy only)
            approx_step: step size for numerical approximation of Jacobian (scipy only)
            opto:        optimization library to use, either "scipy" or "matlab"
        """
        t0 = None
        t1 = None
        t2 = None
        t0 = time.perf_counter() ###################
        if opto.startswith("matlab") and self.matlabEng is None:
            raise ValueError("invalid argument for opto: matlab engine not available")

        funcs = None
        try:
            funcs = GMMMixPLAggregator.mixPLalgorithms[algorithm]
        except KeyError:
            raise ValueError("invalid argument value for algorithm: '" + str(algorithm) + "'")

        # choose constraints for objective function
        Aeq = None
        beq = None
        if algorithm.endswith("uncons"):
            if opto == "scipy":
                raise NotImplementedError("unconstrained optimization with scipy not implemented")
            else: # opto startswith "matlab"
                Aeq = self.Aeq_uncons
                beq = self.beq_uncons
        else: # opto is constrained
            Aeq = self.Aeq
            beq = self.beq

        # compute moment condition values
        moments = None
        mixPLobjective_partial = None
        if opto.startswith("matlab_emp"):
            if true_params is None:
                raise ValueError("invalid value 'true_params=None' when 'opto=" + opto + "''")
            if algorithm.startswith("top3_full"):
                moments = calcMomentsMatlabEmpirical(true_params)
            elif algorithm.startswith("top3_min"):
                moments = calcMomentsMatlabEmpirical_reduced(true_params)
            else:
                raise NotImplementedError("matlab empirical optimization not implemented for moments: '" + algorithm + "''")
        else: # opto is exactly "scipy" or "matlab"
            moments = funcs.calcMoments(rankings)
            # partial function only used for scipy
            mixPLobjective_partial = functools.partial(funcs.mixPLobjective,
                                                       moments=moments
                                                      )

        # generate an initial guess for the optimizer
        params_t0 = np.empty(2*self.m + 1)
        params_t0[0] = np.random.rand()
        params_t0[1:self.m+1] = np.random.dirichlet(np.ones(self.m))
        params_t0[self.m+1:] = np.random.dirichlet(np.ones(self.m))

        # optimization
        res = None
        if opto == "scipy":
            res = scipy.optimize.minimize(mixPLobjective_partial,
                                          params_t0,
                                          method="SLSQP",
                                          bounds=self.bounds_pairs,
                                          constraints=self.cons,
                                          options={
                                            'disp': False,
                                            'maxiter': max_iters,
                                            'ftol': epsilon,
                                            'eps': approx_step
                                            }
                                         )
            res = res.x
        elif opto.startswith("matlab"):
            tolfun = 1e-10
            tolx = 1e-10
            tolcon = 1e-8
            if opto.endswith("_default"): # default tolerances for interior-point
                tolfun = 1e-6
                tolx = 1e-10
                tolcon = 1e-6
            elif opto.endswith("_ultra"): # "optimized" tolerances for interior-point
                tolfun = 1e-13
                tolx = 1e-13
                tolcon = 1e-9

            moments = matlab.double(moments.tolist())
            params_t0 = matlab.double(params_t0.tolist())
            t1 = time.perf_counter() ###################
            res, val, fl = self.matlabEng.optimize("gmm_mixpl_objectives." + funcs.mixPLobjective.__name__,
                                                   moments,
                                                   params_t0,
                                                   self.A,
                                                   self.b,
                                                   Aeq,
                                                   beq,
                                                   self.lb,
                                                   self.ub,
                                                   {"Algorithm": "interior-point",
                                                    "Display": "off",
                                                    "TolFun": tolfun,
                                                    "TolX": tolx,
                                                    "TolCon": tolcon},
                                                   nargout=3
                                                  )
            t2 = time.perf_counter() ###################
            res = np.array(res[0])

        return (res, t1 - t0, t2 - t1)


if __name__ == "__main__":
    import sys
    n = 1
    m = 4
    algo = "top3_full"
    cand_set = np.arange(m)
    gmmagg = GMMMixPLAggregator(cand_set, use_matlab=True)
    wsse_vals = np.empty(1000)
    sse_vals = np.empty(1000)
    ##j = 0
    np.random.seed(0)
    print("i =   ", end='')
    for i in range(1000):
        print("\b"*len(str(i-1)) + str(i), end='')
        sys.stdout.flush()

        params, votes = pl.generate_mix2pl_dataset(n, m, True)

        #momnts = GMMMixPLAggregator.mixPLalgorithms[algo].calcMoments(votes)
        #gmmagg = GMMMixPLAggregator(cand_set, use_matlab=True)

        #print("SciPy Optimize:\n" + "="*20)
        #sol_params = gmmagg.aggregate(votes, algorithm=algo, epsilon=1e-10, max_iters=300, approx_step=1.4901161193847656e-08, opto="scipy")

        #print("Ground-Truth Parameters:\n" + str(params))
        #print("Final Solution:\n" + str(sol_params))
        #print("WSSE:\n" + str(stats.mix2PL_wsse(params, sol_params, m)))
        #print("Ground-Truth Value:\n" + str(GMMMixPLAggregator.mixPLalgorithms[algo].mixPLobjective(params, momnts)))
        #print("Minimum Found:\n" + str(GMMMixPLAggregator.mixPLalgorithms[algo].mixPLobjective(sol_params, momnts)))

        # use empirical ground-truth moment values from here below:
        #momnts = calcMomentsMatlab(params) # top3 full (20 moments)

        sol_params = gmmagg.aggregate(None, algorithm=algo, epsilon=None, max_iters=None, approx_step=None, opto="matlab_emp_ultra", true_params=params)
        wsse_vals[i] = stats.mix2PL_wsse(params, sol_params[0], m)
        sse_vals[i] = stats.mix2PL_sse(params, sol_params[0], m)
        ##if j < 10:
        ##    if wsse_vals[i] >= 1e-03:
        ##        print("\nWSSE:\n" + str(wsse_vals[i]))
        ##        print("Ground-Truth:\n" + str(params))
        ##        print("Solution Found:\n" + str(sol_params))
        ##        print()
        ##        j += 1
        #print("Ground-Truth Parameters:\n" + str(params))
        #print("Final Solution:\n" + str(sol_params))
        #print("WSSE:\n" + str(stats.mix2PL_wsse(params, sol_params, m)))
        #print("Ground-Truth Value:\n" + str(GMMMixPLAggregator.mixPLalgorithms[algo].mixPLobjective(params, momnts)))
        #print("Minimum Found:\n" + str(GMMMixPLAggregator.mixPLalgorithms[algo].mixPLobjective(sol_params, momnts)))

        #algo += "_uncons" # unconstrained optimization

        #print("\n\n\nMatLab Fmincon Unconstrained:\n" + "="*20)
        #sol_params = gmmagg.aggregate(votes, algorithm=algo, epsilon=None, max_iters=None, approx_step=None, opto="matlab", true_params=None)

        #print("Ground-Truth Parameters:\n" + str(params))
        #print("Final Solution:\n" + str(sol_params))
        #print("WSSE:\n" + str(stats.mix2PL_wsse(params, sol_params, m)))
        #print("Ground-Truth Value:\n" + str(GMMMixPLAggregator.mixPLalgorithms[algo].mixPLobjective(params, momnts)))
        #print("Minimum Found:\n" + str(GMMMixPLAggregator.mixPLalgorithms[algo].mixPLobjective(sol_params, momnts)))

    print("WSSE Vals:\n" + str(wsse_vals))
    print("Mean WSSE:\n" + str(np.mean(wsse_vals)))
    print("StdD WSSE:\n" + str(np.std(wsse_vals)))
    print("Median WSSE:\n" + str(np.median(wsse_vals, overwrite_input=True)))
    print()
    print("SSE Vals:\n" + str(sse_vals))
    print("Mean SSE:\n" + str(np.mean(sse_vals)))
    print("StdD SSE:\n" + str(np.std(sse_vals)))
    print("Median SSE:\n" + str(np.median(sse_vals, overwrite_input=True)))
    gmmagg.matlabEng.exit() # cleanup matlab engine properly when done
