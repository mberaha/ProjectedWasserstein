"""
Run the numerical illustration on the set of Dirichlet Process Mixtures
"""
import argparse
import multiprocessing
import numpy as np
import pickle
import time

from copy import deepcopy
from functools import partial
from scipy.integrate import simps
from scipy.stats import norm, gamma, beta, dirichlet
from scipy.interpolate import interp1d, PchipInterpolator

from pwass.distributions import Distribution
from pwass.spline import SplineBasis, MonotoneQuadraticSplineBasis
from pwass.dimsensionality_reduction.geodesic_pca import GeodesicPCA
from pwass.dimsensionality_reduction.nested_pca import NestedPCA
from pwass.dimsensionality_reduction.projected_pca import ProjectedPCA
from pwass.dimsensionality_reduction.simplicial_pca import SimplicialPCA
np.random.seed(20200711)

xgrid = np.linspace(-10, 10, 1000)

def simulate_data(ndata):
    # approximate a DP by truncation
    gamma = 50
    L = 500

    out = []
    for i in range(ndata):
        weights = np.random.dirichlet(np.ones(L) / gamma, 1)
        atoms = np.empty((L, 2))
        atoms[:, 0] = np.random.normal(loc=0.0, scale=2.0, size=L)
        atoms[:, 1] = np.random.uniform(0.5, 2.0, size=L)
        dens_ = norm.pdf(xgrid.reshape(-1, 1), atoms[:, 0], atoms[:, 1])
        dens = np.sum(dens_ * weights, axis=1)
        dens += 1e-5
        totmass = simps(dens, xgrid)
        dens /= totmass
        curr = Distribution()
        curr.init_from_pdf(xgrid, dens)
        out.append(curr)
    return out



def simp_dist(f, pca):
    reduced = pca.transform([f])
    rec = pca.pt_from_proj(reduced) + spca.bary
    rec_pdf = pca.get_pdf(rec)

    qgrid = np.cumsum(f.pdf_eval) * (f.pdf_grid[1] - f.pdf_grid[0])
    orig_qeval = f.pdf_grid

    # reconstruct the INV-CDF
    rec_cdf = np.cumsum(rec_pdf) * (f.pdf_grid[1] - f.pdf_grid[0])

    # adjust for possible flat regions
    keep = np.where(np.diff(rec_cdf) > 1e-10)
    rec_invcdf = PchipInterpolator(
        rec_cdf[keep], f.pdf_grid[keep], extrapolate=False)

    rec_invcdf_eval = rec_invcdf(qgrid)
    nans = np.where(np.isnan(rec_invcdf_eval))[0]
    if len(nans) > 0:
        rec_invcdf_eval[nans[nans < len(rec_invcdf_eval) / 2]] = 0.0
        rec_invcdf_eval[nans[nans > len(rec_invcdf_eval) / 2]] = 1.0

    er = np.sqrt(np.sum(
        (orig_qeval - rec_invcdf_eval)[1:]**2 * np.diff(qgrid)))

    return er


def fit_and_error_simp(data, pca, dim):
    pca.fit(data, dim)
    error = np.mean([simp_dist(d, pca) for d in data])
    print("SIMP dim: {0}, error: {1}".format(dim, error))
    return error


def reconstruction_error(true_distribs, pca):
    proj_coeffs = pca.transform(true_distribs)
    reconstructed_coeffs = pca.pt_from_proj(proj_coeffs) + pca.bary

    mean = 0
    for i in range(len(true_distribs)):
        pt = true_distribs[i].quantile_coeffs
        delta = pt - reconstructed_coeffs[i, :]
        mean += np.sqrt(pca.inner_prod(delta, delta))

    return mean / len(true_distribs)


def fit_and_compute_error(data, pca, dim):
    pca.fit(data, dim)
    error = reconstruction_error(data, pca)
    print("dim: {0}, error: {1}".format(dim, error))
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndata", type=int, default=100)
    parser.add_argument("--nbasis", type=int, default=20)
    parser.add_argument("--ncomp", type=int, default=10)
    parser.add_argument("--nrep", type=int, default=10)
    parser.add_argument("--output", type=str, default="dpm_sim_res.pickle")
    args = parser.parse_args()

    nbasis = args.nbasis
    sbasis = SplineBasis(deg=2, xgrid=xgrid, nbasis=nbasis)

    zero_one_grid = np.linspace(0, 1, 1000)
    wbasis = MonotoneQuadraticSplineBasis(nbasis, zero_one_grid)

    ppca = ProjectedPCA(spline_basis=wbasis, compute_spline=False)
    npca = NestedPCA(spline_basis=wbasis, compute_spline=False)
    gpca = GeodesicPCA(spline_basis=wbasis, compute_spline=False)
    spca = SimplicialPCA(nbasis, spline_basis=sbasis, compute_spline=False)

    dim_range = np.arange(2, args.ncomp + 1, 2)
    print("dim_range: ", dim_range)

    p_errors = np.empty((args.nrep, len(dim_range)))
    n_errors = np.empty((args.nrep, len(dim_range)))
    g_errors = np.empty((args.nrep, len(dim_range)))
    s_errors = np.empty((args.nrep, len(dim_range)))


    for i in range(args.nrep):
        print("********** Running repetition: {0} **********".format(i+1))
        data = simulate_data(args.ndata)
        
        for d in data:
            d.wbasis = wbasis
            d.xbasis = sbasis
            d.compute_clr()
            d.compute_spline_expansions()
        
        for j, dim in enumerate(dim_range):
            print("Running dimension: {0}".format(dim))
            
            print("Simplicial")
            s_errors[i, j] = fit_and_error_simp(data, spca, dim)

            print("Global")
            g_errors[i, j] = fit_and_compute_error(data, gpca, dim)

            print("Projected")
            p_errors[i, j] = fit_and_compute_error(data, ppca, dim)

            print("Nested")
            n_errors[i, j] = fit_and_compute_error(data, npca, dim)

        with open(args.output, "wb") as fp:
            pickle.dump(
                {"p_errors": p_errors, "n_errors": n_errors,
                 "g_errors": g_errors, "s_errors": s_errors},
                fp)
