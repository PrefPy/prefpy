import sys
import time
import numpy as np
import plackettluce as pl
import stats as stats
import gmm_mixpl
#import evbwie as emm
import plot_mixpl_matlab as plot
np.seterr(all='raise')


def print_usage(argv0):
    print("USAGE: python3", argv0, "<# of alternatives> <# of trials> <# of votes start> <# of votes end> <# votes step> <# iters for EMM> <wsse output filename.csv> <time output filename.csv> <plot output filename.png> <gmm t-test output filename.csv> [-U]\n" +
          "  Optional Parameters:\n    -U : use a uniform distribution (default is Dirichlet)\n" +
          "  Notes:\n    Value of \"<# iters for EMM>\" less than 1 will run EMM algorithm to convergence with epsilon=0.0001 and max_iters=100")
    sys.exit()

def main(argv):
    if len(argv) < 11:
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

    # Check files can be written to later:
    wsse_filename = argv[7]
    wsse_file = open(wsse_filename, 'w')
    wsse_file.close()
    time_filename = argv[8]
    time_file = open(time_filename, 'w')
    time_file.close()
    plot_filename = argv[9]
    plot_file = open(plot_filename, 'w')
    plot_file.close()

    # Generate data sets with Dirichlet by default:
    useDirich = True
    if len(argv) > 11 and argv[11] == "-U":
        useDirich = False

    results = np.empty((2, p, 5)) # 2 statistics X p points X 5 observations
    # results[0] is wsse
    # reults[1] is time
    # results[.][h] is experiment h (i.e. h*100 agents)
    # results[.][.][0] is n
    # results[.][.][1] is top2_full GMM (16 moments)
    # results[.][.][2] is top3_full GMM (20 moments)
    # results[.][.][3] is top2_min GMM (12 moments)
    # results[.][.][4] is top3_min GMM (16 moments)
    # results[.][.][5] is EMM

    alts = np.arange(m)

    # initialize the aggregators for each class of algorithm
    gmmagg = gmm_mixpl.GMMMixPLAggregator(alts, use_matlab=True)

    k_n = 0 # experiment index number
    opto_fails = 0
    for n in range(n_init, n_stop + 1, n_step): # for these numbers of agents
        #np.random.seed(0)
        print("n =", n)
        print("i =   ", end='')
        sys.stdout.flush()
        wsse_vals = np.empty((4,t))
        time_vals = np.empty((4,t))
        for i in range(t):
            print("\b"*len(str(i-1)) + str(i), end='')
            sys.stdout.flush()

            while True:
                try:
                    # generate data
                    params, votes = pl.generate_mix2pl_dataset(n, m, useDirich)

                    # MatLab top3_full empirical moments limit
                    time_val = time.perf_counter()
                    soln = gmmagg.aggregate(votes, algorithm="top3_full", epsilon=None, max_iters=None, approx_step=None, opto="matlab3", true_params=params)
                    time_val = time.perf_counter() - time_val
                    wsse_val = stats.mix2PL_wsse(params, soln, m)
                    wsse_vals[0][i] = wsse_val
                    time_vals[0][i] = time_val

                    # MatLab top3_full empirical moments limit unconstrained
                    time_val = time.perf_counter()
                    soln = gmmagg.aggregate(votes, algorithm="top3_full_uncons", epsilon=None, max_iters=None, approx_step=None, opto="matlab3", true_params=params)
                    time_val = time.perf_counter() - time_val
                    wsse_val = stats.mix2PL_wsse(params, soln, m)
                    wsse_vals[1][i] = wsse_val
                    time_vals[1][i] = time_val

                    # MatLab top3_full GMM (20 moments)
                    time_val = time.perf_counter()
                    soln = gmmagg.aggregate(votes, algorithm="top3_full", epsilon=None, max_iters=None, approx_step=None, opto="matlab", true_params=None)
                    time_val = time.perf_counter() - time_val
                    wsse_val = stats.mix2PL_wsse(params, soln, m)
                    wsse_vals[2][i] = wsse_val
                    time_vals[2][i] = time_val

                    # MatLab top3_full GMM (20 moments) unconstrained
                    time_val = time.perf_counter()
                    soln = gmmagg.aggregate(votes, algorithm="top3_full_uncons", epsilon=None, max_iters=None, approx_step=None, opto="matlab", true_params=None)
                    time_val = time.perf_counter() - time_val
                    wsse_val = stats.mix2PL_wsse(params, soln, m)
                    wsse_vals[3][i] = wsse_val
                    time_vals[3][i] = time_val

                except FloatingPointError: # bad data (objective function NaN)
                    opto_fails += 1
                    continue # retry
                except ValueError: # bad data (votes arrays number of dimensions mismatch)
                    opto_fails += 1
                    continue # retry
                break # good data!

        print()
        results[0][k_n][0] = n
        results[0][k_n][1] = np.mean(wsse_vals[0])
        print("Mean WSSE matlab3 (constrained)  = " + str(results[0][k_n][1]))
        print("StdD WSSE matlab3 (constrained)  = " + str(np.std(wsse_vals[0])))
        results[0][k_n][2] = np.mean(wsse_vals[1])
        print("Mean WSSE matlab3 (unconstrained) = " + str(results[0][k_n][2]))
        print("StdD WSSE matlab3 (unconstrained) = " + str(np.std(wsse_vals[1])))
        results[0][k_n][3] = np.mean(wsse_vals[2])
        print("Mean WSSE matlab (constrained) = " + str(results[0][k_n][3]))
        print("StdD WSSE matlab (constrained) = " + str(np.std(wsse_vals[2])))
        results[0][k_n][4] = np.mean(wsse_vals[3])
        print("Mean WSSE matlab (unconstrained) = " + str(results[0][k_n][4]))
        print("StdD WSSE matlab (unconstrained) = " + str(np.std(wsse_vals[3])))

        results[1][k_n][0] = n
        results[1][k_n][1] = np.mean(time_vals[0])
        results[1][k_n][2] = np.mean(time_vals[1])
        results[1][k_n][3] = np.mean(time_vals[2])
        results[1][k_n][4] = np.mean(time_vals[3])

        k_n += 1

    print("Optimization Failures Count: " + str(opto_fails))

    np.savetxt(wsse_filename, results[0], delimiter=',', newline="\r\n")
    np.savetxt(time_filename, results[1], delimiter=',', newline="\r\n")
    plot.plot_wsse_time_data(results[0], results[1], plot_filename)


if __name__ == "__main__":
    main(sys.argv)
