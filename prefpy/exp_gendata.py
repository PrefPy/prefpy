import plackettluce as pl
from egmm_mixpl import *
import numpy as np

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

if __name__ == "__main__":
    m = 4
    n = 10
    #gt, data = pl.generate_mix2pl_dataset(n, m, useDirichlet=False)
    #print(data[0])
    #perms = []
    #for j in range(len(data)):
    #    perms.append(argsort(data[j]))
    perms = np.random.randint(5, size=(n, m))
    #print(perms[0])
    rslt = egmm_mixpl(perms, m, k = 2, itr = 20)
    #gtp = np.zeros((2,m+1),float)
    #gtp[0][0]=gt[0]
    #gtp[1][0]=1-gt[0]
    #gtp[0][1:m+1] = gt[1:m+1]
    #gtp[1][1:m+1] = gt[m+1:2*m+1]
    #diff = gtp-rslt
    #other = np.zeros((2,m+1),float)
    #other[0] = gtp[0] - rslt[1]
    #other[1] = gtp[1] - rslt[0]
    #print("Ground Truth:")
    #print(gtp)
    print("Estimates:")
    print(rslt)
    #mse1 = np.sum(np.square(diff.flat))
    #mse2 = np.sum(np.square(other.flat))
    #print("MSE:",min(mse1,mse2))
