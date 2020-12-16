import logging
import numpy as np
from qpsolvers import solve_qp

from wtda.dimsensionality_reduction.base_pca import PCA
from wtda.spline import MonotoneQuadraticSplineBasis


class ProjectedPCA(object):
    """Functional principal component analysis using the Wasserstein metric

    Parameters
    ----------
    nbasis : int, default -1 (meaning that the spline_basis param will be used)
        Number of quadratic splines to use as basis for the 
        space L^{2\arrowup}([0, 1]), i.e. the space of monothonically
        increasing functions that are also square integrable on [0, 1].
        An element in this space represents a quantile function

    spline_basis : instance of MonotoneQuadraticSplineBasis object, default None
        A MonotoneQuadraticSplineBasis for the interval [0, 1] 

    compute_invcdf: bool, default=True
        If True, the method compute_invcdf() will be called on each
        predictor.
        If this algorithm is used in a cross-validation, it is more
        efficient to call compute_inv_cdf() on the predictors before
        the cross_validation and pass compute_invcdf=False.

    constraints_mat : np.array of shape (nbasis - 1 \times nbasis), default=None
        The constraints_mat is used to project the OLS result, that is an
        element of L^{2}([0, 1]) on L^{2\arrowup}([0, 1]).

    cons_rhs : np.array of shape (nbais -1), default=None
    """

    def __init__(
            self, nbasis=None, spline_basis=None,
            constraints_mat=None,
            cons_rhs=None, compute_invcdf=True):
        if spline_basis is not None:
            self.spline_basis = spline_basis
            self.nbasis = spline_basis.nbasis
        else:
            self.nbasis = nbasis
            self.spline_basis = None

        if constraints_mat is None:
            self.constraints_mat = np.zeros((self.nbasis - 1, self.nbasis))
            for i in range(self.nbasis-1):
                self.constraints_mat[i, i] = 1
                self.constraints_mat[i, i+1] = -1

        
        self.cons_rhs = cons_rhs
        self.compute_invcdf = compute_invcdf

        self.new_metric = None
        self.new_constraints_mat = None
        self.new_cons_rhs = None

    def _initialize(self):
        if self.spline_basis is None:
            self.spline_basis = MonotoneQuadraticSplineBasis(
                self.nbasis, np.linspace(0, 1, 100))
        else:
            self.basis = self.spline_basis.nbasis

        self.spline_basis.eval_metric()
        self.metric_aug = self.spline_basis.metric

    def _process_data(self, functions):
        self.ndata = len(functions)
        self.functions = functions
        if self.compute_invcdf:
            logging.erro("Not implemented yet")

        self.coeff_mat = self.get_spline_mat(functions)
        self.bary = np.mean(self.coeff_mat, axis=0)

    def _finalize(self):
        self.new_constraint_mat = - np.diff(
            self.eig_vecs[:, :self.k], axis=0)
        self.new_cons_rhs = self.cons_rhs
        self.new_metric = np.matmul(
            np.matmul(self.eig_vecs.T, self.metric_aug),
            self.eig_vecs)[:self.k, :self.k]

    def get_spline_mat(self, functions):
        """Stacks all the coefficient of the spline expansions by row
        """
        out = np.zeros((len(functions), self.nbasis))
        for i, f in enumerate(functions):
            out[i, :] = f.quantile_coeffs

        eps = np.ones(out.shape[1]) * 1e-6
        for i in range(out.shape[1]):
            out[:, i] += np.sum(eps[:i])

        return out

    def fit(self, functions, k):
        self.k = k
        self._initialize()
        self._process_data(functions)
        coeff_centered = self.coeff_mat - self.bary

        if self.cons_rhs is None:
            self.cons_rhs = np.diff(self.bary)
            
        try:
            M = np.matmul(np.dot(coeff_centered.T, coeff_centered) +
                          self.metric_aug)
            eig_vals, eig_vecs = np.linalg.eig(M)
        except Exception as e:
            M = np.matmul(np.dot(coeff_centered.T, coeff_centered) +
                        np.eye(self.nbasis) * 1e-4, self.metric_aug)
            eig_vals, eig_vecs = np.linalg.eig(M)

        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)
        eig_vecs = eig_vecs / np.sqrt(
            np.diag(np.matmul(np.matmul(eig_vecs.T, self.metric_aug), eig_vecs)))

        aux = np.argsort(eig_vals)
        self.eig_vals = np.flip(np.sort(eig_vals))
        self.eig_vecs = np.flip(eig_vecs[:, aux], axis=1)
        self.base_change = np.linalg.inv(self.eig_vecs)
        self._finalize()

    def transform(self, functions):
        if self.compute_invcdf:
            logging.error("Not implemented yet")

        X = self.get_spline_mat(functions)

        X_trans = np.matmul(X - self.bary, self.base_change.T)[:, :self.k]
        X_proj = np.zeros_like(X_trans)
        for i in range(X_trans.shape[0]):
            pt = X_trans[i, :]
            pt_proj = self.project_on_sub(pt)
            X_proj[i, :] = pt_proj

        return X_proj

    def project(self, x):
        proj_mat = self.metric_aug * 2
        q = -2 * np.dot(self.metric_aug, x)

        x_hat = solve_qp(proj_mat, q, self.constraints_mat,
                         self.cons_rhs + 1e-5)
        return x_hat

    def project_on_sub(self, x):
        P = self.new_metric * 2
        q = -2 * np.dot(self.new_metric, x)

        x_hat = solve_qp(P, q, self.new_constraint_mat, self.new_cons_rhs)
        return x_hat

    def inner_prod(self, x, y):
        out = np.dot(np.dot(self.metric_aug, x).T, y)
        return out

    def pt_from_proj(self, proj_coord):
        pt = np.dot(proj_coord, self.eig_vecs[:, :self.k].T)
        return pt
