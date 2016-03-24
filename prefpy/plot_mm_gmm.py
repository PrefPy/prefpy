import sys
import numpy as np
import matplotlib.pyplot as plt


def print_usage(argv0):
    print("USAGE: python3", argv0, "<mse csv filename> <ktau csv filename> [output png filename]")
    sys.exit()

def plot_mse_ktau_data(mse_results, ktau_results, output_img_filename=None):
    # Plot data
    fig = plt.figure(num=1, figsize=(1690/96, 615/96), dpi=96)
    plt.subplot(121)
    plt.title("MSE")
    plt.xlabel("n (votes)")
    mm_line, = plt.plot(mse_results.T[0], mse_results.T[1], "bs", label="MM")
    gmm_line, = plt.plot(mse_results.T[0], mse_results.T[2], "g^", label="GMM-F")
    plt.subplot(122)
    plt.title("Kendall Correlation")
    plt.xlabel("n (votes)")
    plt.plot(ktau_results.T[0], ktau_results.T[1], "bs", label="MM")
    plt.plot(ktau_results.T[0], ktau_results.T[2], "g^", label="GMM-F")
    fig.legend([mm_line, gmm_line], ["MM", "GMM-F"], loc="center right")
    if output_img_filename is not None:
        plt.savefig(output_img_filename, dpi=96)
    else:
        plt.show()

def main(argv):
    if len(argv) < 3:
        print("Inavlid number of arguments provided")
        print_usage(argv[0])

    # Load data from file
    mse_results = np.loadtxt(argv[1], delimiter=',')
    ktau_results = np.loadtxt(argv[2], delimiter=',')

    out_img = None
    if len(argv) >= 4:
        out_img = argv[3]

    plot_mse_ktau_data(mse_results, ktau_results, out_img)


if __name__ == "__main__":
    main(sys.argv)
