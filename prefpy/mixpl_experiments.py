import sys
import time
import pickle
import numpy as np
import plackettluce as pl
import stats as stats
import gmm_mixpl
import evbwie as emm
import plot_mixpl as plot
np.seterr(all='raise')


def print_usage(argv0):
    print("USAGE: python3", argv0, "<# of alternatives> <# of trials> <# of votes start> <# of votes end> <# votes step> <# iters for EMM> <# iters for MM> <dataset input filename base> <wsse output filename.csv> <sse output filename.csv> <time output filename.csv> <plot output filename.png> <gmm results output filename.p> <emm results output filename.p> <wsse t-test output filename.csv> <sse t-test output filename.csv>\n" +
          "  Notes:\n    Value of \"<# iters for EMM>\" less than 1 will run EMM algorithm to convergence with epsilon=1e-8 and max_iters=500\n" +
          "    Value of \"<# iters for MM>\" less than 1 will run the M-step MM algorithm to convergence with epsilon=1e-8 and max_iters=50\n" +
          "    All data files read from disk must be CSV format and have the explicit file extension '.csv'\n" +
          "    The dataset base file name given must be suffixed with an underscore, followed by padding zeros, and without the '.csv' extension (e.g. 'mixpl-dataset_000' where there are at most 999 'mixpl-dataset_<number>.csv' files in the same directory)")
    sys.exit()

def main(argv):
    if len(argv) != 16:
        print_usage(argv[0])
    m = int(argv[1]) # number of alternatives
    t = int(argv[2]) # (index + 1) of final dataset (number of datasets if base filename is 0...00)
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

    # Set EMM algorithm iterations & convergence parameters:
    emm_epsilon = None # TODO: add option to specify on command-line
    mm_epsilon = None # TODO: add option to specify on command-line
    emm_iters = int(argv[6])
    if emm_iters < 1:
        emm_epsilon = 1e-8
        emm_iters = 500
    mm_iters = int(argv[7])
    if mm_iters < 1:
        mm_epsilon = 1e-8
        mm_iters = 50

    # read in all data required for experiments
    print("Reading Datasets from Disk...")
    datasets = []
    data_filename_base = argv[8]
    d = int(data_filename_base.split("_")[-1])
    if d < 0:
        print("Error: dataset base file name must not contain a negative number")
        print_usage(argv[0])
    len_d = str(len(data_filename_base.split("_")[-1]))
    data_filename_base = "_".join(data_filename_base.split("_")[:-1])
    for i in range(t):
        infilename = data_filename_base + '_' + ("{0:0" + len_d + "d}").format(i + d) + ".csv"
        infile = open(infilename)
        datasets.append(pl.read_mix2pl_dataset(infile, numVotes=n_stop))

    # Check files can be written to later:
    wsse_filename = argv[9]
    wsse_file = open(wsse_filename, 'w')
    sse_filename = argv[10]
    sse_file = open(sse_filename, 'w')
    time_filename = argv[11]
    time_file = open(time_filename, 'w')
    plot_filename = argv[12]
    plot_file = open(plot_filename, 'w')
    gmm_solns_filename = argv[13]
    gmm_solns_file = open(gmm_solns_filename, 'wb') # writable binary mode
    emm_solns_filename = argv[14]
    emm_solns_file = open(emm_solns_filename, 'wb') # writable binary mode
    ttest_wsse_filename = argv[15]
    ttest_wsse_file = open(ttest_wsse_filename, 'w')
    ttest_sse_filename = argv[16]
    ttest_sse_file = open(ttest_sse_filename, 'w')

    wsse_res = np.empty((p, 3))
    sse_res = np.empty((p, 3))
    time_res = np.empty((p, 5))
    ##wsse_res = np.empty((p, 2))
    ##sse_res = np.empty((p, 2))
    ##time_res = np.empty((p, 4))
    ttest_vals = np.empty((2,p,3)) # 2 t-tests X p points X 3 values (n, mean, std)

    gmm_solns = []
    emm_solns = []

    alts = np.arange(m)

    # initialize the aggregators for each class of algorithm
    print("Initializing Aggregator Classes...")
    gmmagg = gmm_mixpl.GMMMixPLAggregator(alts, use_matlab=True)
    emmagg = emm.EMMMixPLAggregator(alts)

    print("Starting Experiments...")
    k_n = 0 # experiment index number
    for n in range(n_init, n_stop + 1, n_step): # for these numbers of agents
        print("n =", n)
        print("i =   ", end='')
        sys.stdout.flush()

        wsse_vals = np.empty((2,t))
        sse_vals = np.empty((2,t))
        time_vals = np.empty((4,t))
        ##wsse_vals = np.empty((1,t))
        ##sse_vals = np.empty((1,t))
        ##time_vals = np.empty((3,t))
        diff_vals = np.empty((2,t))

        for i in range(t):
            print("\b"*len(str(i-1)) + str(i), end='')
            sys.stdout.flush()

            # get data
            params, votes = datasets[i]
            votes_curr = votes[:n]

            # top3_full GMM (20 moments)
            time_val = time.perf_counter()
            soln, t0, t1 = gmmagg.aggregate(rankings = votes_curr,
                                            ##rankings = None, # for ground-truth empirical limit
                                            algorithm = "top3_full",
                                            epsilon = None,
                                            max_iters = None,
                                            approx_step = None,
                                            opto = "matlab",
                                            ##opto = "matlab_emp", # for ground-truth empirical limit
                                            true_params = None
                                            ##true_params = params # for ground-truth empirical limit
                                           )
            time_val = time.perf_counter() - time_val
            wsse_val = stats.mix2PL_wsse(params, soln, m)
            sse_val = stats.mix2PL_sse(params, soln, m)
            wsse_vals[0][i] = wsse_val
            sse_vals[0][i] = sse_val
            time_vals[0][i] = t0
            time_vals[1][i] = t1
            time_vals[2][i] = time_val
            gmm_result = gmm_mixpl.GMMMixPLResult(num_alts = m,
                                                  num_votes = n,
                                                  ##num_votes = 0, # ground-truth empirical limit
                                                  num_mix = 2,
                                                  true_params = params,
                                                  cond = "top3_full",
                                                  opto = "matlab",
                                                  ##opto = "matlab_emp",  # ground-truth empirical limit
                                                  soln_params = soln,
                                                  momnts_runtime = t0,
                                                  opto_runtime = t1,
                                                  overall_runtime = time_val
                                                 )
            gmm_solns.append(gmm_result)

            # EMM
            time_val = time.perf_counter()
            emm_pi, emm_p, pi_0, p_0 = emmagg.aggregate(votes_curr,
                                                        K=2,
                                                        epsilon=emm_epsilon,
                                                        max_iters=emm_iters,
                                                        epsilon_mm=emm_epsilon,
                                                        max_iters_mm=mm_iters
                                                       )
            time_val = time.perf_counter() - time_val
            soln = np.hstack((emm_pi[0], emm_p[0], emm_p[1]))
            wsse_val = stats.mix2PL_wsse(params, soln, m)
            sse_val = stats.mix2PL_sse(params, soln, m)
            wsse_vals[1][i] = wsse_val
            sse_vals[1][i] = sse_val
            time_vals[3][i] = time_val
            emm_result = emm.EMMMixPLResult(num_alts = m,
                                            num_votes = n,
                                            num_mix = 2,
                                            true_params = params,
                                            epsilon = emm_epsilon,
                                            max_iters = emm_iters,
                                            epsilon_mm = emm_epsilon,
                                            max_iters_mm = mm_iters,
                                            init_guess = np.hstack((
                                                pi_0[0],
                                                p_0[0],
                                                p_0[1]
                                            )),
                                            soln_params = soln,
                                            runtime = time_val
                                           )
            emm_solns.append(emm_result)

            # t-test differences
            diff_vals[0][i] = wsse_vals[1][i] - wsse_vals[0][i]
            diff_vals[1][i] = sse_vals[1][i] - sse_vals[0][i]

        print()
        wsse_res[k_n][0] = n
        wsse_res[k_n][1] = np.mean(wsse_vals[0]) # GMM
        wsse_res[k_n][2] = np.mean(wsse_vals[1]) # EMM

        sse_res[k_n][0] = n
        sse_res[k_n][1] = np.mean(sse_vals[0]) # GMM
        sse_res[k_n][2] = np.mean(sse_vals[1]) # EMM

        time_res[k_n][0] = n
        time_res[k_n][1] = np.mean(time_vals[0]) # GMM t0 (moment-calc)
        time_res[k_n][2] = np.mean(time_vals[1]) # GMM t1 (optimization)
        time_res[k_n][3] = np.mean(time_vals[2]) # GMM overall time
        time_res[k_n][4] = np.mean(time_vals[3]) # EMM time

        # GMM vs EMM WSSE
        ttest_vals[0][k_n][0] = n
        ttest_vals[0][k_n][1] = np.mean(diff_vals[0])
        ttest_vals[0][k_n][2] = np.std(diff_vals[0])

        # GMM vs EMM SSE
        ttest_vals[1][k_n][0] = n
        ttest_vals[1][k_n][1] = np.mean(diff_vals[1])
        ttest_vals[1][k_n][2] = np.std(diff_vals[1])

        # write results intermediately after a full set of trials for each n
        pickle.dump(gmm_solns, gmm_solns_file)
        pickle.dump(emm_solns, emm_solns_file)

        k_n += 1

    pickle.dump(gmm_solns, gmm_solns_file)
    gmm_solns_file.close()
    pickle.dump(emm_solns, emm_solns_file)
    emm_solns_file.close()
    np.savetxt(wsse_filename, wsse_res, delimiter=',', newline="\r\n")
    wsse_file.close()
    np.savetxt(sse_filename, sse_res, delimiter=',', newline="\r\n")
    sse_file.close()
    np.savetxt(time_filename, time_res, delimiter=',', newline="\r\n")
    time_file.close()
    np.savetxt(ttest_wsse_filename, ttest_vals[0], delimiter=',', newline="\r\n")
    ttest_wsse_file.close()
    np.savetxt(ttest_sse_filename, ttest_vals[1], delimiter=',', newline="\r\n")
    ttest_sse_file.close()

    plot.plot_wsse_time_data(str_error_type="MSE",
                             error_results=wsse_res,
                             time_results=time_res,
                             output_img_filename=plot_filename
                            )
    plot_file.close()


if __name__ == "__main__":
    main(sys.argv)
