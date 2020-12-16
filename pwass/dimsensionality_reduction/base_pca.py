import numpy as np

from abc import ABCMeta, abstractmethod
from qpsolvers import solve_qp

from pwass.spline import MonotoneQuadraticSplineBasis

class PCA(metaclass=ABCMeta):
    def __init__(
        self, nbasis=None, spline_basis=None, 
        constraints_mat=None, cons_rhs=None, compute_spline=True):
        if nbasis is None and spline_basis is None:
            logging.error("Not enough arguments for constructor")

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

        else:
            self.constraints_mat = constraints_mat

        self.cons_rhs = cons_rhs
        self.compute_spline = compute_spline

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
    

    def _process_data(self, distribs):
        self.ndata = len(distribs)
        self.distribs = distribs
        if self.compute_spline:
            for d in self.distribs:
                d.wbasis = self.spline_basis
                d.compute_spline_expansions()

        self.coeff_mat = self.get_spline_mat(distribs)
        self.bary = np.mean(self.coeff_mat, axis=0)

    def _finalize(self):
        self.new_constraint_mat = - np.diff(
            self.eig_vecs[:, :self.k], axis=0)
        self.new_cons_rhs = self.cons_rhs
        self.new_metric = np.matmul(
            np.matmul(self.eig_vecs.T, self.metric_aug), 
            self.eig_vecs)[:self.k, :self.k]

    def get_spline_mat(self, distribs):
        """Stacks all the coefficient of the spline expansions by row
        """       
        out = np.zeros((len(distribs), self.nbasis))
        for i, d in enumerate(distribs):
            out[i, :] = d.quantile_coeffs

        eps = np.ones(out.shape[1]) * 1e-6
        for i in range(out.shape[1]):
            out[:, i] += np.sum(eps[:i])

        return out

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

    @abstractmethod
    def fit(self, distributions, k):
        pass

    @abstractmethod
    def transform(self, distributions):
        pass
    
