import numpy as np
import theano.tensor as tt
from theano import shared
from pymc3.distributions.dist_math import normal_lcdf

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
    return N_upper_limit * normal_lcdf(mu, sigma, upper_limit)

def exp_atten_curve(dust_amp, dust_index, lam):
    return np.exp((dust_amp[None]*(lam[:,None,None]/5500.)**dust_index[None]))

def get_atten_curve(dust1, dust2, dust_index, lam, young_age):
    young_correction=(young_age[:,None,None,None]*exp_atten_curve(dust1, dust_index, lam)+1.0)
    old_correction = np.ones_like(young_age)[:,None,None,None]*exp_atten_curve(dust2, dust_index, lam)
    total_correction=young_correction*old_correction
    corr_=[]
    corr_.append(np.ones_like(young_correction))
    corr_.append(young_correction)
    corr_.append(old_correction)
    corr_.append(total_correction)
    corr_=np.asarray(corr_)
    return shared(corr_)
