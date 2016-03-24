import sys
import numpy as np
import scipy
import functools
import plackettluce as pl
import stats

def calcMoments(params): # Full top3 (20 moments)
    alpha = params[0]
    a = params[1:5]
    b = params[5:]
    p1 = alpha*a+(1-alpha)*b
    p21 = alpha*a[0]*a[1:]/(1-a[0])+(1-alpha)*b[0]*b[1:]/(1-b[0])
    p22 = alpha*a[1]*np.hstack((a[0],a[2:]))/(1-a[1])+(1-alpha)*b[1]*np.hstack((b[0],b[2:]))/(1-b[1])
    p23 = alpha*a[2]*np.hstack((a[:2],a[3]))/(1-a[2])+(1-alpha)*b[2]*np.hstack((b[:2],b[3]))/(1-b[2])
    p24 = alpha*a[3]*a[:3]/(1-a[3])+(1-alpha)*b[3]*b[:3]/(1-b[3])
    p3 = np.array([
        alpha*a[0]*a[2]*a[3]/(1-a[2])/(a[0]+a[1])+(1-alpha)*b[0]*b[2]*b[3]/(1-b[2])/(b[0]+b[1]),
        alpha*a[0]*a[1]*a[3]/(1-a[3])/(a[1]+a[2])+(1-alpha)*b[0]*b[1]*b[3]/(1-b[3])/(b[1]+b[2]),
        alpha*a[0]*a[1]*a[2]/(1-a[0])/(a[3]+a[2])+(1-alpha)*b[0]*b[1]*b[2]/(1-b[0])/(b[3]+b[2]),
        alpha*a[2]*a[1]*a[3]/(1-a[1])/(a[0]+a[3])+(1-alpha)*b[2]*b[1]*b[3]/(1-b[1])/(b[0]+b[3])
        ])
    return np.concatenate((p1,p21,p22,p23,p24))

def mixPLobjective(params, moments): # Full top3 (20 moments)
    alpha = params[0]
    a = params[1:5]
    b = params[5:]
    p = moments
    p1 = alpha*a+(1-alpha)*b-p[:4]
    p21 = alpha*a[0]*a[1:]/(1-a[0])+(1-alpha)*b[0]*b[1:]/(1-b[0])-p[4:7]
    p22 = alpha*a[1]*np.hstack((a[0],a[2:]))/(1-a[1])+(1-alpha)*b[1]*np.hstack((b[0],b[2:]))/(1-b[1])-p[7:10]
    p23 = alpha*a[2]*np.hstack((a[:2],a[3]))/(1-a[2])+(1-alpha)*b[2]*np.hstack((b[:2],b[3]))/(1-b[2])-p[10:13]
    p24 = alpha*a[3]*a[:3]/(1-a[3])+(1-alpha)*b[3]*b[:3]/(1-b[3])-p[13:16]
    p3 = np.array([
        alpha*a[0]*a[2]*a[3]/(1-a[2])/(a[0]+a[1])+(1-alpha)*b[0]*b[2]*b[3]/(1-b[2])/(b[0]+b[1])-p[16],
        alpha*a[0]*a[1]*a[3]/(1-a[3])/(a[1]+a[2])+(1-alpha)*b[0]*b[1]*b[3]/(1-b[3])/(b[1]+b[2])-p[17],
        alpha*a[0]*a[1]*a[2]/(1-a[0])/(a[3]+a[2])+(1-alpha)*b[0]*b[1]*b[2]/(1-b[0])/(b[3]+b[2])-p[18],
        alpha*a[2]*a[1]*a[3]/(1-a[1])/(a[0]+a[3])+(1-alpha)*b[2]*b[1]*b[3]/(1-b[1])/(b[0]+b[3])-p[19]
        ])
    allp = np.concatenate((p1,p21,p22,p23,p24))
    return np.sum(allp**2)

if __name__ == "__main__":
    m = 4
    alpha = 0.1
    gamma1 = np.array([0.4, 0.2, 0.2, 0.2])
    gamma2 = np.array([0.1, 0.3, 0.3, 0.3])

    # collect the parameters into one array
    params = np.empty(2*m + 1)
    params[0] = alpha
    params[1:m+1] = gamma1
    params[m+1:] = gamma2

    # compute 16 moment condition values
    moments = calcMoments(params)

    # generate an initial guess for the optimizer
    #params_t0 = np.empty(2*m + 1)
    #params_t0[0] = np.random.rand()
    #params_t0[1:m+1] = np.random.dirichlet(np.ones(m))
    #params_t0[m+1:] = np.random.dirichlet(np.ones(m))
    params_t0 = np.array([0.4030, 0.2347, 0.2551, 0.2551, 0.2551, 0.0593, 0.3135, 0.3135, 0.3135])

    # optimization
    bounds_pairs = [(0.0, 1.0) for i in range(2*m + 1)]
    cons = ({"type": "eq",
             "fun": lambda x: 1 - np.sum(x[1:m+1])},
            {"type": "eq",
             "fun": lambda x: 1 - np.sum(x[m+1:])}
            )
    mixPLobjective_partial = functools.partial(mixPLobjective, moments=moments)
    res = scipy.optimize.minimize(mixPLobjective_partial,
                                  params_t0,
                                  method="SLSQP",
                                  bounds=bounds_pairs,
                                  constraints=cons,
                                  tol=1e-07,
                                  options={
                                    'disp': False,
                                    'maxiter': 1e5,
                                    'ftol': 1e-07,
                                    'eps': 1.4901161193847656e-08
                                    }
                                 )
    print("Ground-Truth Parameters:\n" + str(params))
    print("True Minimum:\n" + str(mixPLobjective(params, moments)))
    if res.success:
        print("Success!\nMinimum Found:\n" + str(res.fun))
    else:
        print("Failed!")
    print("Minimizing Value:\n" + str(res.x))
    print("MSE:\n" + str(stats.mix2PL_mse(params, res.x, m)))
