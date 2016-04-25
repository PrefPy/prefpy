import sys
import numpy as np
import matplotlib.pyplot as plt


def print_usage(argv0):
    print("USAGE: python3", argv0, "<wsse csv filename> <time csv filename> [output png filename]")
    sys.exit()

def plot_wsse_time_data(wsse_results, time_results, output_img_filename=None):
    # Plot data
    fig = plt.figure(num=1, figsize=(800/96, 500/96), dpi=96)
    #plt.subplot(121)
    ax = plt.title("WMSE")
    plt.xlabel("n (votes)")
    line1, = plt.plot(wsse_results.T[0], wsse_results.T[1], "bs", label="GT-Cons")
    line2, = plt.plot(wsse_results.T[0], wsse_results.T[2], "g^", label="GT-Uncons")
    line3, = plt.plot(wsse_results.T[0], wsse_results.T[3], "ro", label="Cons")
    line4, = plt.plot(wsse_results.T[0], wsse_results.T[4], "m+", label="Uncons")
    #plt.subplot(122)
    #plt.title("Time (seconds)")
    #plt.xlabel("n (votes)")
    #plt.plot(time_results.T[0], time_results.T[1], "bs", label="Matlab-GT-Cons")
    #plt.plot(time_results.T[0], time_results.T[2], "g^", label="Matlab-GT-Uncon")
    #plt.plot(time_results.T[0], time_results.T[3], "ro", label="Matlab-Cons")
    #plt.plot(time_results.T[0], time_results.T[3], "m+", label="Matlab-Uncon")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.legend([line1, line2, line3, line4], ["GT-Cons", "GT-Uncons", "Cons", "Uncons"], loc="center right", bbox_to_anchor=(1, 0.5))
    if output_img_filename is not None:
        plt.savefig(output_img_filename, dpi=96)
    else:
        plt.show()

def main(argv):
    if len(argv) < 3:
        print("Inavlid number of arguments provided")
        print_usage(argv[0])

    # Load data from file
    wsse_results = np.loadtxt(argv[1], delimiter=',')
    time_results = np.loadtxt(argv[2], delimiter=',')

    out_img = None
    if len(argv) >= 4:
        out_img = argv[3]

    plot_wsse_time_data(wsse_results, time_results, out_img)


if __name__ == "__main__":
    main(sys.argv)
