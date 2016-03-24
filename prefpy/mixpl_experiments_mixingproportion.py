import sys
import numpy as np
import plackettluce as pl
import stats
import gmm_mixpl
import plot_mixpl_mixingproportion as plot
np.seterr(all='raise')


def print_usage(argv0):
    print("USAGE: python3", argv0, "<# of alternatives> <# of trials> <alpha start & step value> <number of steps> <wsse output filename.csv> <sse output filename.csv> <plot output filename.png> [-U]\n" +
          "  Optional Parameters:\n    -U : use a uniform distribution (default is Dirichlet)\n")
    sys.exit()

def main(argv):
    if len(argv) < 8:
        print_usage(argv[0])
    m = int(argv[1]) # number of alternatives
    t = int(argv[2]) # number of trials
    a_init = float(argv[3]) # starting and stepping value for alpha
    p = int(argv[4]) # number of times to increment by a_init

    # Check files can be written to later:
    wsse_filename = argv[5]
    wsse_file = open(wsse_filename, 'w')
    wsse_file.close()
    sse_filename = argv[6]
    sse_file = open(sse_filename, 'w')
    sse_file.close()
    plot_filename = argv[7]
    plot_file = open(plot_filename, 'w')
    plot_file.close()

    # Generate data sets with Dirichlet by default:
    useDirich = True
    if len(argv) > 8 and argv[8] == "-U":
        useDirich = False

    results = np.empty((2, p, 3)) # 2 statistics X p points X 3 observations

    alts = np.arange(m)

    # initialize the aggregators for each class of algorithm
    gmmagg = gmm_mixpl.GMMMixPLAggregator(alts, use_matlab=True)

    opto_fails = 0
    for n in range(p):
        alpha = a_init * (n + 1)
        np.random.seed(0)
        print("n =", n)
        print("i =   ", end='')
        sys.stdout.flush()
        wsse_vals = np.empty(t)
        sse_vals = np.empty(t)
        for i in range(t):
            print("\b"*len(str(i-1)) + str(i), end='')
            sys.stdout.flush()

            while True:
                try:
                    # generate ground-truths
                    gamma1 = np.random.dirichlet(np.ones(m))
                    gamma2 = np.random.dirichlet(np.ones(m))
                    params = np.hstack((alpha, gamma1, gamma2))

                    # MatLab top3_full GMM (20 moments)
                    soln = gmmagg.aggregate(rankings=None,
                                            algorithm="top3_full",
                                            epsilon=None,
                                            max_iters=None,
                                            approx_step=None,
                                            opto="matlab3",
                                            true_params=params
                                           )
                    wsse_val = stats.mix2PL_wsse(params, soln, m)
                    sse_val = stats.mix2PL_sse(params, soln, m)
                    wsse_vals[i] = wsse_val
                    sse_vals[i] = sse_val

                except FloatingPointError: # bad data (objective function NaN)
                    opto_fails += 1
                    continue # retry
                except ValueError: # bad data (votes arrays number of dimensions mismatch)
                    opto_fails += 1
                    continue # retry
                break # good data!

        print()
        # store WSSE values
        results[0][n][0] = alpha
        results[0][n][1] = np.mean(wsse_vals)
        results[0][n][2] = np.std(wsse_vals)

        # store SSE values
        results[1][n][0] = alpha
        results[1][n][1] = np.mean(sse_vals)
        results[1][n][2] = np.std(sse_vals)

    print("Optimization Failures Count: " + str(opto_fails))

    np.savetxt(wsse_filename, results[0], delimiter=',', newline="\r\n")
    np.savetxt(sse_filename, results[1], delimiter=',', newline="\r\n")
    plot.plot_wsse_sse_data(results[0], results[1], plot_filename)


if __name__ == "__main__":
    main(sys.argv)
