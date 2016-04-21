import sys
import time
import pickle
import numpy as np
import plackettluce as pl
import stats as stats
import evbwie as emm
import plot_mixpl_emm1 as plot
np.seterr(all='raise')


def print_usage(argv0):
    print("USAGE: python3", argv0, "<# of alternatives> <# of trials> <# of votes start> <# of votes end> <# votes step> <# iters for EMM> <# iters for MM> <dataset input filename base> <wsse output filename.csv> <sse output filename.csv> <time output filename.csv> <plot output filename.png> <emm results output filename.p>\n" +
          "    Notes:\n    All data files read from disk must be CSV format and have the explicit file extension '.csv'" +
          "    The dataset base file name given must be suffixed with an underscore, followed by padding zeros, and without the '.csv' extension (e.g. 'mixpl-dataset_000' where there are at most 999 'mixpl-dataset_<number>.csv' files in the same directory)")
    sys.exit()

def main(argv):
    if len(argv) != 14:
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
        mm_iters = 10

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
    emm_solns_filename = argv[13]
    emm_solns_file = open(emm_solns_filename, 'wb') # writable binary mode

    # open previous experiments results files
    orig_error_results = np.loadtxt("../../MixPL_WsseTimeExperiments/mse_mixPL_04-alts_1000-trials_20.csv", delimiter=',')
    orig_time_results = np.loadtxt("../../MixPL_WsseTimeExperiments/time_mixPL_04-alts_1000-trials_20.csv", delimiter=',')

    #orig_error_results = np.loadtxt("../../MixPL_WsseTimeExperiments/mse_mixPL_04-alts_1000-trials_20_trunc1.csv", delimiter=',')
    #orig_time_results = np.loadtxt("../../MixPL_WsseTimeExperiments/time_mixPL_04-alts_1000-trials_20_trunc1.csv", delimiter=',')

    #orig_error_results = np.loadtxt("../../MixPL_WsseTimeExperiments/mse_mixPL_04-alts_1000-trials_20_trunc2.csv", delimiter=',')
    #orig_time_results = np.loadtxt("../../MixPL_WsseTimeExperiments/time_mixPL_04-alts_1000-trials_20_trunc2.csv", delimiter=',')

    wsse_res = np.empty((p, 2))
    sse_res = np.empty((p, 2))
    time_res = np.empty((p, 2))

    emm_solns = []

    alts = np.arange(m)

    # initialize the aggregators for each class of algorithm
    print("Initializing Aggregator Classes...")
    emmagg = emm.EMMMixPLAggregator(alts)

    print("Starting Experiments...")
    k_n = 0 # experiment index number
    for n in range(n_init, n_stop + 1, n_step): # for these numbers of agents
        print("n =", n)
        print("i =   ", end='')
        sys.stdout.flush()

        wsse_vals = np.empty((1,t))
        sse_vals = np.empty((1,t))
        time_vals = np.empty((1,t))

        for i in range(t):
            print("\b"*len(str(i-1)) + str(i), end='')
            sys.stdout.flush()

            # get data
            params, votes = datasets[i]
            votes_curr = votes[:n]

            # EMM
            time_val = time.perf_counter()
            emm_pi, emm_p, pi_0, p_0 = emmagg.aggregate(votes_curr,
                                                        K=2,
                                                        epsilon=emm_epsilon,
                                                        max_iters=emm_iters,
                                                        epsilon_mm=mm_epsilon,
                                                        max_iters_mm=mm_iters
                                                       )
            time_val = time.perf_counter() - time_val
            soln = np.hstack((emm_pi[0], emm_p[0], emm_p[1]))
            wsse_val = stats.mix2PL_wsse(params, soln, m)
            sse_val = stats.mix2PL_sse(params, soln, m)
            wsse_vals[0][i] = wsse_val
            sse_vals[0][i] = sse_val
            time_vals[0][i] = time_val
            emm_result = emm.EMMMixPLResult(num_alts = m,
                                            num_votes = n,
                                            num_mix = 2,
                                            true_params = params,
                                            epsilon = emm_epsilon,
                                            max_iters = emm_iters,
                                            epsilon_mm = mm_epsilon,
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


        print()
        wsse_res[k_n][0] = n
        wsse_res[k_n][1] = np.mean(wsse_vals[0]) # EMM

        sse_res[k_n][0] = n
        sse_res[k_n][1] = np.mean(sse_vals[0]) # EMM

        time_res[k_n][0] = n
        time_res[k_n][1] = np.mean(time_vals[0]) # EMM

        # write results intermediately after a full set of trials for each n
        pickle.dump(emm_solns, emm_solns_file)

        k_n += 1

    pickle.dump(emm_solns, emm_solns_file)
    emm_solns_file.close()
    np.savetxt(wsse_filename, wsse_res, delimiter=',', newline="\r\n")
    wsse_file.close()
    np.savetxt(sse_filename, sse_res, delimiter=',', newline="\r\n")
    sse_file.close()
    np.savetxt(time_filename, time_res, delimiter=',', newline="\r\n")
    time_file.close()

    plot.plot_error_time_data(str_error_type="MSE",
                              error_results=sse_res,
                              time_results=time_res,
                              orig_error_results=orig_error_results,
                              orig_time_results=orig_time_results,
                              output_img_filename=plot_filename
                             )
    plot_file.close()


if __name__ == "__main__":
    main(sys.argv)
