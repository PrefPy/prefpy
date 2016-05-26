# Implementation of algorithm (2) from
# Exploring Voting Blocs Within the Irish Electorate:
# A Mixture Modeling Approach by Gormley and Murphy, 2008

import numpy as np
from . import aggregate
from . import plackettluce as pl
from . import stats


#deprecated
class _EMMMixPLResult_legacy:
    """
    Description:
        Legacy class used to generate EMM solutions files for all experiments
        prior to publication of Zhao, Piech, & Xia (2016).  All new code should
        use the new class.
    """
    def __init__(self, num_alts, num_votes, num_mix, true_params, epsilon, max_iters, epsilon_mm, max_iters_mm, init_guess, soln_params, runtime):
        self.num_alts = num_alts
        self.num_votes = num_votes
        self.num_mix = num_mix
        self.true_params = true_params
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.epsilon_mm = epsilon_mm
        self.max_iters_mm = max_iters_mm
        self.init_guess = init_guess
        self.soln_params = soln_params
        self.runtime = runtime

class EMMMixPLResult:
    """
    Description:
        Class used to store important values pertaining to an execution of the
        EMM algorithm and its return.  This class contains values to assist
        further investigations of solutions provided by this method.
    """
    def __init__(self, num_alts, num_votes, num_mix, true_params, epsilon, epsilon_mm, iters, init_guess, soln_params, runtime):
        self.num_alts = num_alts
        self.num_votes = num_votes
        self.num_mix = num_mix
        self.true_params = true_params
        self.epsilon = epsilon
        self.epsilon_mm = epsilon_mm
        self.iters = iters
        self.init_guess = init_guess
        self.soln_params = soln_params
        self.runtime = runtime

class EMMMixPLAggregator(aggregate.RankAggregator):

    def c(x_i, j):
        try:
            return x_i[j]
        except IndexError:
            return -1

    def f(x_i, p):
        prod = 1
        for t in range(len(x_i)):
            denom_sum = 0
            for s in range(t, len(p)):
                denom_sum += p[EMMMixPLAggregator.c(x_i, s)]
            prod *= p[EMMMixPLAggregator.c(x_i, t)] / denom_sum
        return prod

    def indic(j, x_i, s):
        flag = j == EMMMixPLAggregator.c(x_i, s)
        if flag:
            return 1
        else:
            return 0

    def delta(x_i, j, s, N):
        """ delta_i_j_s """
        flag = j == EMMMixPLAggregator.c(x_i, s)
        if flag and s < len(x_i):
            return 1
        elif s == N:
            found_equal = False
            for l in range(len(x_i)):
                if j == EMMMixPLAggregator.c(x_i, l):
                    found_equal = True
                    break
            if not found_equal:
                return 1
        return 0

    def omega(k, j, z, x):
        """ omega_k_j """
        sum_out = 0
        for i in range(len(x)):
            sum_in = 0
            for t in range(len(x[i])):
                sum_in += z[i][k] * EMMMixPLAggregator.indic(j, x[i], t)
            sum_out += sum_in
        return sum_out

    def aggregate(self, rankings, K, epsilon, epsilon_mm, iters):
        """
        Description:
            Takes in a set of rankings and computes the model
            parameters for a mixture of Plackett-Luce models.
        Parameters:
            rankings:   set of rankings to aggregate
            K:          number of mixture components to compute
            epsilon:    convergence condition threshold value for overall EM algorithm
            epsilon_mm: convergence condition threshold value for MM algorithm
            iters:      dict, iterations configuration for EM and MM algorithms
        """
        x = rankings # shorter pseudonym for voting data
        self.n = len(rankings) # number of votes

        # "fixed" iterations type variables
        outer_iters = None
        inner_iters = None
        inner_range = None

        # Additional "scaling" iterations type variables
        inner_iters_base = None
        scaling_divisor = None

        # Additional "total" iterations type variables
        total_iters = None
        isIncremented = False

        if "type" not in iters:
            raise ValueError("iters dict must contain key \"type\"")
        iters_type = iters["type"]
        if iters_type == "fixed":
            outer_iters = iters["em_iters"]
            inner_iters = iters["mm_iters"]
        elif iters_type == "scaling":
            outer_iters = iters["em_iters"]
            inner_iters_base = iters["mm_iters_base"]
            scaling_divisor = iters["scaling_divisor"]
        elif iters_type == "total":
            total_iters = iters["total_iters"]
            outer_iters = iters["em_iters"]
            inner_iters = total_iters // outer_iters
        else:
            raise ValueError("iters dict value for key \"type\" is invalid: " + str(iters_type))

        # pre-compute the delta values
        delta_i_j_s = np.empty((self.n, self.m, self.m + 1))
        for i in range(self.n):
            for j in range(self.m):
                for s in range(self.m + 1):
                    delta_i_j_s[i][j][s] = EMMMixPLAggregator.delta(x[i], j, s, self.m)

        # generate initial values for p and pi:
        p_h0 = np.random.rand(K, self.m)
        p_h0 /= np.sum(p_h0, axis=1, keepdims=True)

        pi_h0 =  np.random.rand(K)
        pi_h0 /= np.sum(pi_h0)

        p_h = np.copy(p_h0)
        pi_h = np.copy(pi_h0)

        for g in range(outer_iters):

            p_h1 = np.empty((K, self.m))
            pi_h1 = np.empty(K)
            z_h1 = np.empty((self.n, K))

            # E-Step:
            self._EStep(K, x, z_h1, pi_h, p_h)

            # M-Step:
            if iters_type == "fixed":
                inner_range = range(inner_iters)
            elif iters_type == "total" and not isIncremented:
                test = (g + 1) * inner_iters + (outer_iters - g - 1) * (inner_iters + 1)
                if test < total_iters:
                    inner_iters += 1
                    isIncremented = True
                inner_range = range(inner_iters)
            elif iters_type == "scaling":
                inner_range = range(int(g/scaling_divisor) + inner_iters_base)

            for l in inner_range:
                self._MStep(l, K, x, z_h1, pi_h1, p_h1, p_h, delta_i_j_s)

                if (epsilon_mm != None and
                    np.all(np.absolute(p_h1 - p_h) < epsilon_mm)):
                        break

                p_h = np.copy(p_h1) # deep copy p for next MM iteration
                # pi does not change across MM iterations, no copy needed

            if (epsilon != None and
                np.all(np.absolute(p_h1 - p_h) < epsilon) and
                np.all(np.absolute(pi_h1 - pi_h) < epsilon)):
                    break

            # remember that assignments below are references only, not copies
            p_h = p_h1
            pi_h = pi_h1

        return (pi_h1, p_h1, pi_h0, p_h0)


    def _EStep(self, K, x, z_h1, pi_h, p_h):
        """
        Description:
            Internal function for computing the E-Step of the EMM algorithm.
        """
        # E-Step:
        for i in range(self.n):
            for k in range(K):
                denom_sum = 0
                for k2 in range(K):
                    denom_sum += pi_h[k2] * EMMMixPLAggregator.f(x[i], p_h[k2])
                z_h1[i][k] = (pi_h[k] * EMMMixPLAggregator.f(x[i], p_h[k])) / denom_sum


    def _MStep(self, l, K, x, z_h1, pi_h1, p_h1, p_h, delta_i_j_s):
        """
        Description:
            Internal function for computing the M-Step of the EMM algorithm,
            which is itself an MM algorithm.
        """
        for k in range(K):
            normconst = 0
            if l == 0: # only need to compute pi at first MM iteration
                pi_h1[k] = np.sum(z_h1.T[k]) / len(z_h1)
            for j in range(self.m):
                omega_k_j = EMMMixPLAggregator.omega(k, j, z_h1, x) # numerator
                denom_sum = 0
                for i in range(self.n):
                    sum1 = 0
                    for t in range(len(x[i])):
                        sum2 = 0
                        sum3 = 0
                        for s in range(t, self.m):
                            sum2 += p_h[k][EMMMixPLAggregator.c(x[i], s)]
                        for s in range(t, self.m + 1):
                            sum3 += delta_i_j_s[i][j][s]
                        sum1 += z_h1[i][k] * (sum2 ** -1) * sum3
                    denom_sum += sum1
                p_h1[k][j] = omega_k_j / denom_sum
                normconst += p_h1[k][j]
            for j in range(self.m):
                p_h1[k][j] /= normconst


def main():
    n = 100
    m = 4
    k = 2
    cand_set = np.arange(m)
    #np.random.seed(0)
    params, votes = pl.generate_mix2pl_dataset(n, m, useDirichlet=True)
    print("Ground-Truth Parameters:\n" + str(params))
    print("EMM Algorithm:")

    emmagg = EMMMixPLAggregator(cand_set)
    pi, p, pi_h0, p_h0 = emmagg.aggregate(votes,
                                          K=2,
                                          epsilon=None,
                                          epsilon_mm=None,
                                          iters={"type" : "fixed",
                                                 "em_iters": 20,
                                                 "mm_iters": 5}
                                         )

    sol_params = np.empty(2*m+1)
    sol_params[0] = pi[0]
    sol_params[1:m+1] = p[0]
    sol_params[m+1:] = p[1]

    print("Ground-Truth Parameters:\n" + str(params))
    print("Final Solution:\n" + str(sol_params))
    print("\t\"1 - alpha\" = " + str(pi[1]))
    print("MSE:\n" + str(stats.mix2PL_sse(params, sol_params, m)))

if __name__ == "__main__":
    main()
