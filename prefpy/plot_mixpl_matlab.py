import sys
import numpy as np
import matplotlib.pyplot as plt


def print_usage(argv0):
    print("USAGE: python3", argv0, "<wsse csv filename> <time csv filename> [output png filename]")
    sys.exit()

def plot_wsse_time_data(wsse_results, time_results, output_img_filename=None):
    # Plot data
    fig = plt.figure(num=1, figsize=(1690/96, 615/96), dpi=96)
    plt.subplot(121)
    plt.title("Mean Weighted SSE")
    plt.xlabel("n (votes)")
    line1, = plt.plot(wsse_results.T[0], wsse_results.T[1], "bs", label="Matlab-GT-Cons")
    line2, = plt.plot(wsse_results.T[0], wsse_results.T[2], "g^", label="Matlab-GT-Uncon")
    line3, = plt.plot(wsse_results.T[0], wsse_results.T[3], "ro", label="Matlab-Cons")
    line4, = plt.plot(wsse_results.T[0], wsse_results.T[4], "m+", label="Matlab-Uncon")
    plt.subplot(122)
    plt.title("Time (seconds)")
    plt.xlabel("n (votes)")
    plt.plot(time_results.T[0], time_results.T[1], "bs", label="Matlab-GT-Cons")
    plt.plot(time_results.T[0], time_results.T[2], "g^", label="Matlab-GT-Uncon")
    plt.plot(time_results.T[0], time_results.T[3], "ro", label="Matlab-Cons")
    plt.plot(time_results.T[0], time_results.T[3], "m+", label="Matlab-Uncon")
    fig.legend([line1, line2, line3, line4], ["Matlab-GT-Cons", "Matlab-GT-Uncon", "Matlab-Cons", "Matlab-Uncon"], loc="center right")
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
