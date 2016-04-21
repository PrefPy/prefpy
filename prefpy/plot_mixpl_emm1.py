import numpy as np
import matplotlib.pyplot as plt

def plot_error_time_data(str_error_type,          # string, title of the error measure in use
                         error_results,           # error results
                         time_results,            # time results
                         orig_error_results,      # original GMM vs EMM error results
                         orig_time_results,       # original GMM vs EMM time results
                         output_img_filename=None # output png filename
                        ):
    # Plot data
    fig = plt.figure(num=1, figsize=(1400/96, 500/96), dpi=96)
    plt.subplot(121)
    plt.title(str_error_type)
    plt.xlabel("n (votes)")
    gmm_line, = plt.plot(orig_error_results.T[0], orig_error_results.T[1], "bs", label="GMM")
    emm_line, = plt.plot(orig_error_results.T[0], orig_error_results.T[2], "g^", label="EMM-500")
    emm_new1, = plt.plot(error_results.T[0], error_results.T[1], "mH", label="EMM-10-5")
    plt.subplot(122)
    plt.title("Time (seconds)")
    plt.xlabel("n (votes)")
    plt.plot(orig_time_results.T[0], orig_time_results.T[3], "bs", label="GMM")
    plt.plot(orig_time_results.T[0], orig_time_results.T[4], "g^", label="EMM-500")
    plt.plot(time_results.T[0], time_results.T[1], "mH", label="EMM-10-5")
    fig.legend([gmm_line, emm_line, emm_new1], ["GMM", "EMM-500", "EMM-10-5"], loc="center right")
    if output_img_filename is not None:
        plt.savefig(output_img_filename, dpi=96)
    else:
        plt.show()
