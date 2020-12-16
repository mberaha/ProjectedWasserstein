""" Utilities for automatically running an analysis on a dataset """
import logging
import joblib
import numpy as np
import pickle

from joblib import Parallel, delayed, effective_n_jobs
from itertools import combinations, product

from wtda.t_function import TFunction


def loadData(filename):
    with open(filename, "rb") as fp:
        data = pickle.load(fp)

    func_eval = data["func_eval"]
    grid = data.get("grid", None)
    response = data["labels"]
    if grid is None:
        grid = np.arange(func_eval.shape[1])

    ymin = np.min(func_eval)
    ymax = np.max(func_eval)
    sublevel_grid = np.linspace(ymin, ymax, 500)
    tfuncs = np.array([TFunction(func_eval[i, :], grid, sublevel_grid)
              for i in range(func_eval.shape[0])])

    return tfuncs, response


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
    