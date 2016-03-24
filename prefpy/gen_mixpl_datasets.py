import sys
import numpy as np
import plackettluce as pl

def print_usage(argv0):
    print("USAGE: python3 " + argv0 + " <num alts> <num votes> <num datasets> <dataset(s) base filename.csv>")
    sys.exit()

def main(argv):
    if len(argv) != 5:
        print_usage(argv[0])
    m = int(argv[1])
    n = int(argv[2])
    len_d = str(len(argv[3]))
    d = int(argv[3])
    filename_base = argv[4]

    print("i =   ", end='')
    for i in range(d):
        print("\b"*len(str(i-1)) + str(i), end='')
        sys.stdout.flush()
        outfilename = filename_base + '_' + ("{0:0" + len_d + "d}").format(i) + ".csv"
        outfile = open(outfilename, 'w')
        pl._generate_mix2pl_dataset(n, m, outfile, True)
        outfile.close()

if __name__ == "__main__":
    main(sys.argv)
