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
    return tt.exp((dust_amp.dimshuffle('x',0,1)*(lam.dimshuffle(0,'x','x')/5500.)**dust_index.dimshuffle('x',0,1)))

def get_atten_curve(dust1, dust2, dust_index, ext_dust1, ext_dust2, ext_dust_index, lam, young_age, sh):
    young_undone=young_age.dimshuffle(0,'x','x','x')*exp_atten_curve(dust1, dust_index, lam)+(1.0-young_age).dimshuffle(0,'x','x','x')*tt.ones((sh[1],sh[2],sh[3],sh[4]))
    old_undone = tt.ones_like(young_age).dimshuffle(0,'x','x','x')*exp_atten_curve(dust2, dust_index, lam)
    young_correction=young_age.dimshuffle(0,'x','x','x')*exp_atten_curve(-ext_dust1, ext_dust_index, lam)+(1.0-young_age).dimshuffle(0,'x','x','x')*tt.ones((sh[1],sh[2],sh[3],sh[4]))
    old_correction=tt.ones_like(young_age).dimshuffle(0,'x','x','x')*exp_atten_curve(-ext_dust2, ext_dust_index, lam)
    total_undone=young_undone*old_undone
    total_correction = young_correction*old_correction
    corr_=tt.zeros(sh)
    corr_=tt.set_subtensor(corr_[0], total_correction*total_undone)
    corr_=tt.set_subtensor(corr_[1], old_correction*total_undone)
    corr_=tt.set_subtensor(corr_[2], young_correction*total_undone)
    corr_=tt.set_subtensor(corr_[3], total_undone)
    return corr_
