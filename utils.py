import numpy as np
import theano.tensor as tt
from pymc3.distributions.dist_math import normal_lccdf

def wquantile(data, weights, quantile):
    # Code from wquantile package
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    Pn = (Sn-0.5*sorted_weights)/Sn[-1]
    return np.interp(quantile, Pn, sorted_data)

def stick_breaking(beta, M):
    portion_remaining = beta*tt.concatenate([tt.ones((M,1)),tt.cumprod(1.0-beta,axis=1)[:,:-1]],axis=1)
    portion_remaining=portion_remaining/tt.sum(portion_remaining, axis=1, keepdims=True)
    return portion_remaining

def upper_limit_likelihood(mu, sigma, N_upper_limit, upper_limit):
    return N_upper_limit * normal_lccdf(mu, sigma, upper_limit)
