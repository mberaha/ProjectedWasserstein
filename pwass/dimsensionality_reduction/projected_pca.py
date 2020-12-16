import numpy as np
from qpsolvers import solve_qp

from pwass.dimsensionality_reduction.base_pca import PCA
from pwass.spline import MonotoneQuadraticSplineBasis


class ProjectedPCA(PCA):
    """Functional principal component analysis using the Wasserstein metric
    """

    def __init__(
            self, nbasis=None, spline_basis=None,
            constraints_mat=None,
            cons_rhs=None, compute_spline=True):
        super().__init__(nbasis, spline_basis, constraints_mat,
                         cons_rhs, compute_spline)

    def fit(self, distribs, k):
        self.k = k
        self._initialize()
        self._process_data(distribs)
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

    def transform(self, distribs):
        if self.compute_spline:
            for d in distribs:
                d.wbasis = self.spline_basis
                d.compute_spline_expansions()

        X = self.get_spline_mat(distribs)

        X_trans = np.matmul(X - self.bary, self.base_change.T)[:, :self.k]
        X_proj = np.zeros_like(X_trans)
        for i in range(X_trans.shape[0]):
            pt = X_trans[i, :]
            pt_proj = self.project_on_sub(pt)
            X_proj[i, :] = pt_proj

        return X_proj

