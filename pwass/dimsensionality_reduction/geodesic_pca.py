import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from pyomo.core.util import sum_product, quicksum

from pwass.dimsensionality_reduction.base_pca import PCA
from pwass.spline import MonotoneQuadraticSplineBasis


class GeodesicPCA(PCA):
    def __init__(self, nbasis=None, spline_basis=None, constraints_mat=None,
                 cons_rhs=None, compute_spline=True):

        super().__init__(nbasis, spline_basis, constraints_mat,
                         cons_rhs, compute_spline)

    def fit(self, distributions, k):
        self.k = k
        self._initialize()
        self._process_data(distributions)

        if self.cons_rhs is None:
            self.cons_rhs = np.diff(self.bary)

        coeffs_centered = self.coeff_mat - self.bary
        M = np.matmul(coeffs_centered, self.metric_aug)

        aux_ = np.dot(coeffs_centered.T, coeffs_centered)
        M_ = np.matmul(aux_ + np.eye(self.nbasis) * 1e-4, self.metric_aug)

        (eig_val, eig_vecs) = np.linalg.eig(M_)
        eig_val = np.real(eig_val)
        eig_vecs = np.real(eig_vecs)

        eig_vecs = eig_vecs / \
            np.sqrt(
                np.diag(np.matmul(np.matmul(eig_vecs.T, self.metric_aug), eig_vecs)))

        aux = np.argsort(eig_val)
        self.eig_val = np.flip(np.sort(eig_val))
        self.initialization = np.flip(eig_vecs[:, aux], axis=1)

        self.eig_vecs = np.zeros((self.nbasis, self.nbasis))
        self.coords = np.ones((self.ndata, self.nbasis))

        self.eig_vecs, self.coords = self.find_components(
            self.k, coeffs_centered)
        self._finalize()

    def find_components(self, n, coeffs_centered):
        def norm(model):
            out = 0
            for i in range(self.ndata):
                x_i = np.copy(coeffs_centered[i, :])
                pt_i = np.copy(self.coeff_mat[i, :])
                center = np.copy(self.bary)
                PT = []
                for s in range(self.nbasis):
                    vector = 0
                    for k in range(model.ncomp):
                        vector += model.lamb[i, k]*model.w[s, k]
                    PT.append(center[s] + vector)

                for h in range(model.nvar):
                    for s in range(model.nvar):
                        out += (PT[h]-pt_i[h]) * self.metric_aug[h, s] * (
                            PT[s]-pt_i[s])
            return out


        model = pyo.ConcreteModel()
        model.nvar = self.nbasis
        model.ncomp = n
        model.npoints = self.ndata

        model.w = pyo.Var(
            np.arange(model.nvar), np.arange(model.ncomp), domain=pyo.Reals,
            initialize=lambda m, i, j: self.initialization[i, j])

        model.lamb = pyo.Var(np.arange(model.npoints), np.arange(
            model.ncomp), domain=pyo.Reals, initialize=1)

        model.obj = pyo.Objective(rule=norm, sense=pyo.minimize)
        model.costr = pyo.ConstraintList()

        # costraint ||w||_E=1
        for k in range(model.ncomp):
            aux = 0
            for i in range(model.nvar):
                aux += self.metric_aug[i, i]*model.w[i, k]**2
                for j in range(i):
                    aux += 2*model.w[i, k]*model.w[j, k]*self.metric_aug[i, j]

            model.costr.add(aux == 1)

        # monothonicity contstraint
        for i in range(self.ndata):
            center = np.copy(self.bary)
            PT = []
            for s in range(self.nbasis):
                vector = 0
                for k in range(model.ncomp):
                    vector += model.lamb[i, k]*model.w[s, k]
                PT.append(center[s] + vector)

            for j in range(self.nbasis-1):
                costr = sum([self.constraints_mat[j, s]*PT[s] 
                             for s in range(self.nbasis)])
                model.costr.add(costr <= 0)

        # orthogonality
        if model.ncomp > 1:
            for k in range(model.ncomp):
                for h in range(k):
                    angle = 0
                    for s in range(model.nvar):
                        for t in range(model.nvar):
                            angle += model.w[s, k] * model.w[t, h] * \
                                self.metric_aug[s, t]
                    model.costr.add(angle == 0)

        solver = pyo.SolverFactory('ipopt')

        S = solver.solve(model)
        cost = model.obj()

        w_eval = np.ones((model.nvar, model.ncomp))
        for key, val in model.w.extract_values().items():
            w_eval[key] = val

        lamb_eval = np.ones((model.npoints, model.ncomp))
        for key, val in model.lamb.extract_values().items():
            lamb_eval[key] = val

        return w_eval, lamb_eval

    def transform(self, distribs):
        if self.compute_spline:
            for d in distribs:
                d.wbasis = self.spline_basis
                d.compute_spline_expansions()

        X = self.get_spline_mat(distribs)
        coords = []
        for npt in range(len(distribs)):
            pt = X[npt, :] - self.bary
            coord = [self.inner_prod(pt, self.eig_vecs[:, k])
                     for k in range(self.k)]
            coords.append(np.array(coord))

        X_trans = np.vstack(coords)
        X_proj = np.zeros_like(X_trans)

        self.X_trans = X_trans
        for i in range(X_trans.shape[0]):
            pt = X_trans[i, :]
            pt_proj = self.project_on_sub(pt)
            X_proj[i, :] = pt_proj

        return X_proj
