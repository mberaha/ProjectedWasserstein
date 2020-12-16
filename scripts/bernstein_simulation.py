"""
Run the numerical illustration on the set of Dirichlet Process Mixtures
"""
import argparse
import multiprocessing
import numpy as np
import pickle
import time

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
np.random.seed(12346)

xgrid = np.linspace(0, 1, 500)


def simulate_data(ndata):
    L = 500
    beta_dens = np.zeros((L, len(xgrid)))
    for j in range(L):
        beta_dens[j, :] = beta.pdf(xgrid, j + 1, L - j)

    out = []
    for i in range(ndata):
        ws = dirichlet.rvs(np.ones(L) * 0.1)[0]
        pdf = np.sum(beta_dens * ws[:, np.newaxis], axis=0)
        curr = Distribution()
        curr.init_from_pdf(xgrid, pdf)
        out.append(curr)
    return out


def w_dist(func, pca):
    proj_coeffs = pca.transform([func])
    reconstructed_coeffs = pca.pt_from_proj(proj_coeffs) + pca.bary

    qgrid = np.cumsum(func.pdf_eval) * (func.pdf_grid[1] - func.pdf_grid[0])
    orig_qeval = func.pdf_grid

    # reconstruct the INV-CDF
    rec_qeval = wbasis.eval_spline(reconstructed_coeffs, qgrid)[0]
    er = np.sqrt(
        np.sum((orig_qeval - rec_qeval)[1:] ** 2 * np.diff(qgrid)))
    return er


def simp_dist(f, pca):
    reduced = pca.transform([f])
    rec = pca.pt_from_proj(reduced) + pca.bary
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


def fit_and_error_w(data, pca, dim):
    start = time.time()
    pca.fit(data, dim)
    end = time.time()
    error = np.mean([w_dist(d, pca) for d in data])
    # print("WASS, dim: {0}, err: {1}".format(dim, error))
    return error, end - start


def fit_and_error_simp(data, pca, dim):
    start = time.time()
    pca.fit(data, dim)
    end = time.time()
    error = np.mean([simp_dist(d, pca) for d in data])
    # print("SIMP, dim: {0}, err: {1}".format(dim, error))
    return error, end - start


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndata", type=int, default=100)
    parser.add_argument("--nrep", type=int, default=10)
    parser.add_argument("--output", type=str, default="bernstein_sim_res.pickle")
    args = parser.parse_args()

    basis_range = [5, 10, 15, 25, 40]
    dim_range = [2, 5, 10]

    w_errors = np.empty((args.nrep, len(dim_range), len(basis_range)))
    s_errors = np.empty((args.nrep, len(dim_range), len(basis_range)))

    for i in range(args.nrep):
        print("********** Running repetition: {0} **********".format(i+1))
        data = simulate_data(args.ndata)

        for k, nbasis in enumerate(basis_range):
            print("Running basis: {0}".format(nbasis))
            zero_one_grid = np.linspace(0, 1, 100)
            wbasis = MonotoneQuadraticSplineBasis(nbasis, zero_one_grid)
            sbasis = SplineBasis(deg=2, xgrid=xgrid, nbasis=nbasis)

            ppca = ProjectedPCA(spline_basis=wbasis, compute_spline=False)
            spca = SimplicialPCA(
                nbasis, spline_basis=sbasis, compute_spline=False)

            for d in data:
                d.wbasis = wbasis
                d.xbasis = sbasis
                d.compute_clr()
                d.compute_spline_expansions()

            for j, dim in enumerate(dim_range):
                # print("Running dimension: {0}".format(dim))

                if nbasis >= dim:
                    # print("Simplicial")
                    err, t = fit_and_error_simp(data, spca, dim)
                    s_errors[i, j, k] = err

                    # print("Projected")
                    err, t = fit_and_error_w(data, ppca, dim)
                    w_errors[i, j, k] = err
                else:
                    s_errors[i, j, k] = np.nan
                    w_errors[i, j, k] = np.nan



        with open(args.output, "wb") as fp:
            pickle.dump(
                {"w_errors": w_errors, "s_errors": s_errors}, fp)
