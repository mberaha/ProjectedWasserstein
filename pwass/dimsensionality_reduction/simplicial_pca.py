import numpy as np
from scipy.integrate import simps

from pwass.dimsensionality_reduction.base_pca import PCA
from pwass.spline import SplineBasis


class SimplicialPCA(object):
    def __init__(self, nbasis, k=3, compute_spline=True, spline_basis=None):
        self.nbasis = nbasis
        self.k = k
        self.remove_last_col = False
        self.compute_spline = compute_spline
        self.spline_basis = spline_basis

    @staticmethod
    def clr(f_eval, grid):
        log_f = np.log(f_eval + 1e-30)
        out = log_f - simps(log_f / (grid[-1] - grid[0]), grid)
        return out

    @staticmethod
    def inv_clr(f_eval, grid):
        out = np.exp(f_eval)
        den = simps(out, grid)
        return out / den

    def fit(self, distribs, k):
        """
        All the distributions must be defined on the same grid
        """
        # TODO (@MARIO): check that the grid is the same for all
        # distributions 
        if self.spline_basis is None:
            self.spline_basis = SplineBasis(
                self.k, xgrid=distribs[0].pdf_grid, nbasis=self.nbasis)
                
        self.metric = self.spline_basis.metric
        self._process_data(distribs)
        self.k = k

        coeff_centered = self.coeff_mat - self.bary

        M = np.matmul(np.dot(coeff_centered.T, coeff_centered) + 
                      np.eye(self.nbasis) * 1e-4, self.metric)
        eig_vals, eig_vecs = np.linalg.eig(M)
        eig_vals = np.real(eig_vals) + 1e-6
        eig_vecs = np.real(eig_vecs) + 1e-6
        eig_vecs = eig_vecs / np.sqrt(
            np.diag(np.matmul(np.matmul(eig_vecs.T, self.metric), eig_vecs)))

        aux = np.argsort(eig_vals)
        self.eig_vals = np.flip(np.sort(eig_vals))
        self.eig_vecs = np.flip(eig_vecs[:, aux], axis=1)
        self.base_change = np.linalg.inv(self.eig_vecs)

    def transform(self, distribs):
        X = self.get_spline_mat(distribs)
        X_trans = np.matmul(X - self.bary, self.base_change.T)[:, :self.k]
        return X_trans

    def pt_from_proj(self, proj_coord):
        pt = np.dot(proj_coord, self.eig_vecs[:, :self.k].T)
        return pt

    def get_pdf(self, proj_coeffs):        
        func = self.spline_basis.eval_spline(proj_coeffs, self.pdf_grid)
        return self.inv_clr(func, self.pdf_grid)
        
    def _process_data(self, distribs):
        self.pdf_grid = distribs[0].pdf_grid
        self.ndata = len(distribs)
        self.distribs = distribs
        self.coeff_mat = self.get_spline_mat(distribs)
        self.bary = np.mean(self.coeff_mat, axis=0)


    def get_spline_mat(self, distribs):
        out = np.zeros((len(distribs), self.nbasis))
        for i, d in enumerate(distribs):
            if self.compute_spline:
                out[i, :] = self.spline_basis.get_spline_expansion(
                    self.clr(d.pdf_eval, d.pdf_grid))
            else:
                out[i, :] = d.clr_coeffs

        return out

        
