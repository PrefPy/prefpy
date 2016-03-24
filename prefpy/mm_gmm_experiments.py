import sys
import numpy as np
import scipy.stats
import plackettluce as pl
import stats as stats
import mmgbtl as mm
import gmmra as gmm
import plot_mm_gmm as plot


def print_usage(argv0):
    print("USAGE: python3", argv0, "<# of alternatives> <# of trials> <# of votes start> <# of votes end> <# votes step> <# iters for MM> <mse output filename.csv> <ktau output filename.csv> <plot output filename.png> [-U]\n" +
          "  Optional Parameters:\n    -U : use a uniform distribution (default is Dirichlet)\n" +
          "  Notes:\n    Value of \"<# iters for MM>\" less than 1 will run MM algorithm to convergence with epsilon=0.0001 and max_iters=100")
    sys.exit()

def main(argv):
    if len(argv) < 10:
        print_usage(argv[0])
    m = int(argv[1]) # number of alternatives
    t = int(argv[2]) # number of trials
    n_init = int(argv[3]) # initial experiment number of votes
    if not n_init > 0:
        print("Error: Starting number of votes must be greater than 0")
        print_usage(argv[0])
    n_stop = int(argv[4]) # final experiment number of votes
    if not n_stop > n_init:
        print("Error: Final number of votes must be greater than starting number of votes")
        print_usage(argv[0])
    n_step = int(argv[5]) # number of votes to increment by each time
    if not n_step > 0:
        print("Error: Step number of votes must be greater than 0")
        print_usage(argv[0])
    elif (n_stop - n_init) < n_step or (n_stop - n_init) % n_step != 0:
        print("Warning: Step number of votes doesn't fit range")
    p = ((n_stop - n_init) // n_step) + 1 # always positive and >= 1 by above

    # Set MM algorithm iterations & convergence parameters:
    mm_epsilon = None
    mm_iters = int(argv[6])
    if mm_iters < 1:
        mm_epsilon = 0.0001
        mm_iters = 100

    # Check files can be written to later:
    mse_filename = argv[7]
    mse_file = open(mse_filename, 'w')
    mse_file.close()
    ktau_filename = argv[8]
    ktau_file = open(ktau_filename, 'w')
    ktau_file.close()
    plot_filename = argv[9]
    plot_file = open(plot_filename, 'w')
    plot_file.close()

    # Generate data sets with Dirichlet by default:
    useDirich = True
    if len(argv) > 10 and argv[10] == "-U":
        useDirich = False

    results = np.empty((2, p, 3)) # 2 statistics X p points X 3 observations
    # results[0] is mse
    # reults[1] is kendall
    # results[.][h] is experiment h (i.e. h*100 agents)
    # results[.][.][0] is n
    # results[.][.][1] is MM
    # results[.][.][2] is GMM

    alts = [i for i in range(m)]
    mmagg = mm.MMPLAggregator(alts)
    gmmagg = gmm.GMMPLAggregator(alts)
    k_n = 0 # experiment index number
    for n in range(n_init, n_stop + 1, n_step): # for these numbers of agents
        print("n =", n)
        print("i =   ", end='')
        sys.stdout.flush()
        mse_vals = np.empty((2,t))
        ktau_vals = np.empty((2,t))
        for i in range(t):
            print("\b"*len(str(i-1)) + str(i), end='')
            sys.stdout.flush()
            gamma, votes = pl.generate_pl_dataset(n, m, useDirich) # create data set
            gamma_mm = mmagg.aggregate(votes, mm_epsilon, mm_iters) # no epsilon, but 10 iterations exactly
            gamma_gmm = gmmagg.aggregate(votes) # full breaking
            mse_mm = stats.mse(gamma, gamma_mm) # calc MSE for MM
            mse_gmm = stats.mse(gamma, gamma_gmm) # calc MSE for GMM
            ktau_mm = scipy.stats.kendalltau(gamma, gamma_mm) # calc Kendall for MM
            ktau_gmm = scipy.stats.kendalltau(gamma, gamma_gmm) # calc Kendall for GMM
            mse_vals[0][i] = mse_mm
            mse_vals[1][i] = mse_gmm
            ktau_vals[0][i] = ktau_mm[0]
            ktau_vals[1][i] = ktau_gmm[0]

        print()
        results[0][k_n][0] = n
        results[0][k_n][1] = np.mean(mse_vals[0])
        results[0][k_n][2] = np.mean(mse_vals[1])
        results[1][k_n][0] = n
        results[1][k_n][1] = np.mean(ktau_vals[0])
        results[1][k_n][2] = np.mean(ktau_vals[1])
        k_n += 1

    np.savetxt(mse_filename, results[0], delimiter=',', newline="\r\n")
    np.savetxt(ktau_filename, results[1], delimiter=',', newline="\r\n")
    plot.plot_mse_ktau_data(results[0], results[1], plot_filename)


if __name__ == "__main__":
    main(sys.argv)
