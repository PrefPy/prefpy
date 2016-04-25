import sys
import numpy as np
import matplotlib.pyplot as plt


def print_usage(argv0):
    print("USAGE: python3", argv0, "<error csv filename> <time csv filename> [output png filename]")
    sys.exit()

def plot_error_time_data(str_error_type,
                        error_results,
                        time_results,
                        output_img_filename=None
                       ):
    # Plot data
    fig = plt.figure(num=1, figsize=(1350/96, 500/96), dpi=96)
    plt.subplot(121)
    plt.title(str_error_type)
    plt.xlabel("n (votes)")
    #top2full_line, = plt.plot(error_results.T[0], error_results.T[1], "m+", label="GMM-top2-16")
    #top3full_line, = plt.plot(error_results.T[0], error_results.T[2], "ro", label="GMM-top3-20")
    #top2min_line, = plt.plot(error_results.T[0], error_results.T[3], "yd", label="GMM-top2-12")
    top3full_line, = plt.plot(error_results.T[0], error_results.T[1], "bs", label="GMM")
    emm_line, = plt.plot(error_results.T[0], error_results.T[2], "g^", label="EMM")
    ##gmmEmp_line = plt.hlines(0.007016424717135926435, error_results[0], error_results[-1], colors='r', label="GMM-Emp") # this is the "SSE" empirical minimum limit (not "weighted SSE")
    plt.subplot(122)
    plt.title("Time (seconds)")
    plt.xlabel("n (votes)")
    #plt.plot(time_results.T[0], time_results.T[1], "m+", label="GMM-top2-16")
    #plt.plot(time_results.T[0], time_results.T[2], "ro", label="GMM-top3-20")
    #plt.plot(time_results.T[0], time_results.T[3], "yd", label="GMM-top2-12")
    plt.plot(time_results.T[0], time_results.T[3], "bs", label="GMM") # GMM total overall time
    plt.plot(time_results.T[0], time_results.T[4], "g^", label="EMM") # EMM total overall time
    ##momntCalc_line, = plt.plot(time_results.T[0], time_results.T[1], "ro", label="GMM-Moments") # GMM moment value calc time
    ##optoCalc_line, = plt.plot(time_results.T[0], time_results.T[2], "yd", label="GMM-Opt") # GMM optimization time
    fig.legend([top3full_line, emm_line], ["GMM", "EMM"], loc="center right")
    ##fig.legend([top3full_line, gmmEmp_line, emm_line, momntCalc_line, optoCalc_line], ["GMM", "GMM-Emp", "EMM", "GMM-Moments", "GMM-Opt"], loc="center right")
    ##fig.legend([top3full_line, emm_line, momntCalc_line, optoCalc_line], ["GMM", "EMM", "GMM-Moments", "GMM-Opt"], loc="center right")
    ##fig.legend([top3full_line, gmmEmp_line, momntCalc_line, optoCalc_line], ["GMM", "GMM-Emp", "GMM-Moments", "GMM-Opt"], loc="center right")
    ##fig.legend([top3full_line, momntCalc_line, optoCalc_line], ["GMM", "GMM-Moments", "GMM-Opt"], loc="center right")
    if output_img_filename is not None:
        plt.savefig(output_img_filename, dpi=96)
    else:
        plt.show()

def main(argv):
    if len(argv) < 3:
        print("Inavlid number of arguments provided")
        print_usage(argv[0])

    # Load data from file
    error_results = np.loadtxt(argv[1], delimiter=',')
    time_results = np.loadtxt(argv[2], delimiter=',')

    out_img = None
    if len(argv) >= 4:
        out_img = argv[3]

    plot_error_time_data(error_results, time_results, out_img)


if __name__ == "__main__":
    main(sys.argv)
