import sys
import numpy as np
import matplotlib.pyplot as plt


def print_usage(argv0):
    print("USAGE: python3", argv0, "<wsse csv filename> <sse csv filename> [output plot filename.png]")
    sys.exit()

def plot_wsse_sse_data(wsse_results, sse_results, output_img_filename=None):
    # Plot data
    fig = plt.figure(num=1, figsize=(1690/96, 615/96), dpi=96)
    plt.subplot(121)
    plt.title("Mean Weighted SSE")
    plt.xlabel("Mixing Proportion (alpha)")
    top3full_line, = plt.plot(wsse_results.T[0], wsse_results.T[1], "g^", label="Matlab-Optimized")
    plt.subplot(122)
    plt.title("Mean SSE")
    plt.xlabel("Mixing Proportion (alpha)")
    plt.plot(sse_results.T[0], sse_results.T[1], "g^", label="Matlab-Optimized")
    fig.legend([top3full_line], ["Matlab-Optimized"], loc="center right")
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
    sse_results = np.loadtxt(argv[2], delimiter=',')

    out_img = None
    if len(argv) >= 4:
        out_img = argv[3]

    plot_wsse_sse_data(wsse_results, sse_results, out_img)


if __name__ == "__main__":
    main(sys.argv)
