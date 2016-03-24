import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plackettluce as pl
import gmm_mixpl
import stats


def print_usage(argv0):
    print("USAGE: python3", argv0, "<# of alternatives> <# of trials> <alpha start> <alpha end> <alpha step> <dataset input filename base> <plot base filename> <gmm results output filename.p>\n" +
          "  Notes:\n    All data files read from disk must be CSV format and have the explicit file extension '.csv'" +
          "    The dataset base file name given must be suffixed with an underscore, followed by padding zeros, and without the '.csv' extension (e.g. 'mixpl-dataset_000' where there are at most 999 'mixpl-dataset_<number>.csv' files in the same directory)")
    sys.exit()

def plot_wsse_dist_data(wsse_dist_data, output_img_filename, alpha_str):
    # Plot data
    fig = plt.figure(num=1, figsize=(845/96, 615/96), dpi=96)
    plt.title("Weighted SSE for alpha in " + alpha_str)
    plt.xlabel("Absolute Distance Between PL Support Vectors")
    plt.plot(wsse_dist_data.T[0], wsse_dist_data.T[1], "bs", label="GMM")
    plt.savefig(output_img_filename, dpi=96)

def main(argv):
    if len(argv) != 9:
        print_usage(argv[0])
    m = int(argv[1]) # number of alternatives
    t = int(argv[2]) # number of trials / datasets
    a_init = float(argv[3]) # initial value for alpha
    a_stop = float(argv[4]) # final value for alpha
    a_step = float(argv[5]) # iteration step size for alpha
    data_filename_base = argv[6] # input data filename
    plot_filename_base = argv[7] # output plots filenames
    solns_filename = argv[8]
    solns_file = open(solns_filename, 'wb') # writable binary mode

    print("Reading Datasets from Disk...")
    datasets = []
    len_d = str(len(data_filename_base.split("_")[-1]))
    data_filename_base = "_".join(data_filename_base.split("_")[:-1])
    for i in range(t):
        infilename = data_filename_base + '_' + ("{0:0" + len_d + "d}").format(i) + ".csv"
        infile = open(infilename)
        datasets.append(pl.read_mix2pl_dataset(infile, numVotes=0)[0])

    solns = [] # store all GMM Results

    alts = np.arange(m)
    # initialize the aggregators for each class of algorithm
    print("Initializing GMM Aggregator Class...")
    gmmagg = gmm_mixpl.GMMMixPLAggregator(alts, use_matlab=True)

    print("Starting Experiments...")
    results = []
    # new experiments for all alphas in range:
    print("i =   ", end='')
    for i in range(t):
        print("\b"*len(str(i-1)) + str(i), end='')
        sys.stdout.flush()
        params = datasets[i]

        if params[0] >= a_init and params[0] <= a_stop:
            dist = np.sum(np.abs(params[1:m+1] - params[m+1:]))
            if dist > 0.1:
                time_val = time.perf_counter()
                soln = gmmagg.aggregate(rankings=None, algorithm="top3_full", epsilon=None, max_iters=None, approx_step=None, opto="matlab_emp", true_params=params)
                time_val = time.perf_counter() - time_val
                wsse = stats.mix2PL_wsse(params, soln, m)
                gmm_result = gmm_mixpl.GMMMixPLResult(m, 0, params, "top3_full", "matlab_emp", soln, time_val)
                solns.append(gmm_result)
                results.append([dist, wsse])
            else:
                continue
        else:
            continue

    print("Number of Results to Plot:", len(results))
    if len(results) > 0:
        plot_wsse_dist_data(np.asarray(results), plot_filename_base, '[' + str(a_init) + ", " + str(a_stop) + ']')
        numHi = 0
        for j in range(len(results)):
            if results[j][1] >= 0.02:
                numHi += 1
        print("Results with WSSE >= 0.02:", numHi)
    # end of new experiments

    # old experiments for each interval of alpha vals:
    #alpha = a_init
    #while alpha <= a_stop:
    #    print("alpha =", alpha)
    #    print("i =   ", end='')
    #    sys.stdout.flush()
    #    results = []
    #    numRes = 0
    #    for i in range(t):
    #        print("\b"*len(str(i-1)) + str(i), end='')
    #        sys.stdout.flush()
    #        params = datasets[i]
    #
    #        if params[0] >= alpha and params[0] < alpha + a_step:
    #            dist = np.sum(np.abs(params[1:m+1] - params[m+1:]))
    #            if dist > 0.1:
    #                soln = gmmagg.aggregate(rankings=None, algorithm="top3_full", epsilon=None, max_iters=None, approx_step=None, opto="matlab4", true_params=params)
    #                wsse = stats.mix2PL_wsse(params, soln, m)
    #                results.append([dist, wsse])
    #                numRes += 1
    #            else:
    #                continue
    #        else:
    #            continue
    #    if len(results) > 0:
    #        print("Plotting", numRes, "Results for Alpha=", alpha)
    #        out_filename = plot_filename_base + str(alpha) + ".png"
    #        plot_wsse_dist_data(np.asarray(results), out_filename, str(alpha))
    #    alpha = alpha + a_step

    pickle.dump(solns, solns_file)
    solns_file.close()


if __name__ == "__main__":
    main(sys.argv)
