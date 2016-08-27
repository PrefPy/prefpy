from generate import *
from likelihood_rum import *
from GaussianRUMGMM import *
#from scipy.stats import norm

if __name__ == '__main__':

    m = 4
    n = 1000
    ntrials = 100
    sse = 0

    for t in range(0, ntrials):
        print("Trial: ", t)
        Params = GenerateRUMParameters(m, "normal")
        Data = GenerateRUMData(Params,m,n,"normal")
        GroundTruth = Params["Mean"]
        GroundTruth = GroundTruth - GroundTruth.min()
        mu = GMMGaussianRUM(Data, m, n, itr=1)
        mu = mu - mu.min()
        se = np.square(GroundTruth - mu[0]).sum()
        print("Ground Truth: ", GroundTruth)
        print("Estimate: ", mu)
        print("Error: ", se)
        sse += se

    mse = sse/ntrials
    print("MSE: ", mse)
