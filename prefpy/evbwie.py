# Implementation of algorithm (2) from
# Exploring Voting Blocs Within the Irish Electorate:
# A Mixture Modeling Approach by Gormley and Murphy, 2008

import numpy as np
import aggregate
import plackettluce as pl
import stats


class EMMMixPLResult:
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

    def aggregate(self, rankings, K, epsilon, max_iters, epsilon_mm, max_iters_mm):
        x = rankings # shorter pseudonym for voting data
        self.n = len(rankings) # number of votes

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

        for g in range(max_iters):

            p_h1 = np.empty((K, self.m))
            pi_h1 = np.empty(K)
            z_h1 = np.empty((self.n, K))

            # E-Step:
            for i in range(self.n):
                for k in range(K):
                    denom_sum = 0
                    for k2 in range(K):
                        denom_sum += pi_h[k2] * EMMMixPLAggregator.f(x[i], p_h[k2])
                    z_h1[i][k] = (pi_h[k] * EMMMixPLAggregator.f(x[i], p_h[k])) / denom_sum

            # M-Step:
            #for l in range(max_iters_mm):
            for l in range(int(g/50) + 5):
                for k in range(K):
                    normconst = 0
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

                if (epsilon_mm != None and
                    np.all(np.absolute(p_h1 - p_h) < epsilon_mm) and
                    np.all(np.absolute(pi_h1 - pi_h) < epsilon_mm)):
                        break

            if (epsilon != None and
                np.all(np.absolute(p_h1 - p_h) < epsilon) and
                np.all(np.absolute(pi_h1 - pi_h) < epsilon)):
                    break

            p_h = p_h1
            pi_h = pi_h1

        return (pi_h1, p_h1, pi_h0, p_h0)

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
    pi, p = emmagg.aggregate(votes, K=2, epsilon=1e-8, max_iters=1000, epsilon_mm=1e-8, max_iters_mm=10)

    sol_params = np.empty(2*m+1)
    sol_params[0] = pi[0]
    sol_params[1:m+1] = p[0]
    sol_params[m+1:] = p[1]

    print("Ground-Truth Parameters:\n" + str(params))
    print("Final Solution:\n" + str(sol_params))
    print("\t\"1 - alpha\" = " + str(pi[1]))
    print("WSSE:\n" + str(stats.mix2PL_wsse(params, sol_params, m)))

if __name__ == "__main__":
    main()
