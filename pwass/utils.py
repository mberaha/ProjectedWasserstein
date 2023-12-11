""" Utilities for automatically running an analysis on a dataset """
import logging
import joblib
import numpy as np
import pickle

from joblib import Parallel, delayed, effective_n_jobs
from itertools import combinations, product
from scipy.signal import argrelextrema, savgol_filter
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz, simps



def pushforward_density(transport, unif_grid):
    """Computes the density of T(U) when U is the uniform distribution"""
    
    def get_local_extrema(f):
        """Returns a list of indices of local extrema"""
        maxima = argrelextrema(f, np.greater)[0]
        minima = argrelextrema(f, np.less)[0]
        return np.sort(np.concatenate([maxima, minima]))
    
    eval_grid = np.linspace(np.min(transport), np.max(transport), len(unif_grid))
    const = np.where(np.diff(transport) < 1e-5)
    
    extrema = get_local_extrema(transport)
    keep = np.delete(np.arange(len(unif_grid)), extrema)
    
    delta = unif_grid[1] - unif_grid[0]
    dt = -savgol_filter(transport[keep], window_length=5, polyorder=3, deriv=1)
    dens = interp1d(transport[keep], 1.0 / np.abs(dt), bounds_error=False)(eval_grid)
    dens[np.isnan(dens)] = 0.0    
    for ex in extrema:   
        dens[ex] = dens[ex-1]
    
    dens /= simps(dens, eval_grid) 
    return dens, eval_grid


def _dist_wrapper(dist_func, *args, **kwargs):
    """Write in-place to a slice of a distance matrix"""
    return seq_pairwise_dist(dist_func, *args, **kwargs)



def parallel_pairwise_dist(func, X, Y=None, njobs=-1):
    if njobs < 1:
        njobs = joblib.cpu_count() + njobs

    if Y is None:
        Y = X

    fd = delayed(_dist_wrapper)
    out = Parallel(n_jobs=njobs)(
        fd(func, X, Y[s])
        for s in gen_even_slices(len(Y), effective_n_jobs(njobs)))
    return np.hstack(out)
    

def gen_even_slices(n, n_packs, n_samples=None):
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1"
                         % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end


def seq_pairwise_dist(distfunc, X, Y=None):
    if Y is None:
        out = np.zeros((len(X), len(X)))
        for i, j in combinations(range(len(X)), 2):
            out[i, j] = distfunc(X[i], X[j])

        out += out.T
    else:
        out = np.zeros((len(X), len(Y)))
        for i, j in product(range(len(X)), range(len(Y))):
            out[i, j] = distfunc(X[i], Y[j])

    return out
    