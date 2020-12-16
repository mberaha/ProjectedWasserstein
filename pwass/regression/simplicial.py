import numpy as np
from sklearn.base import BaseEstimator

from pwass.spline import SplineBasis
from pwass.distributions import Distribution


class SimpliciadDistribOnDistrib(BaseEstimator):
    def __init__(self, fit_intercept=True, nbasis=-1, spline_basis=None,
                 compute_spline=True):
        self.fit_intercept = fit_intercept
        self.nbasis = nbasis
        self.spline_basis = spline_basis
        self.compute_spline = compute_spline

    def _initialize(self, X):
        if self.spline_basis is None:
            self.spline_basis = SplineBasis(
                2, nbasis=self.nbasis, xgrid=X[0].pdf_grid)
        else:
            self.nbasis = self.spline_basis.nbasis

        self.spline_basis.eval_metric()

    def fit(self, X, Y):
        self._initialize(X)
        self.n_samples = len(X)
        self.X = X
        self.Y = Y

        if self.compute_spline:
            for x in self.X:
                x.xbasis = self.spline_basis
                x.compute_spline_expansions()

            for y in self.Y:
                y.xbasis = self.spline_basis
                y.compute_spline_expansions()

        self.Xmat = self.get_spline_mat(self.X)
        self.Ymat = self.get_spline_mat(self.Y)
        if self.fit_intercept:
            self.Xmat = np.hstack(
                [np.ones(self.n_samples).reshape(-1, 1), self.Xmat])

        self.beta = np.linalg.solve(
            np.matmul(self.Xmat.T, self.Xmat),
            np.matmul(self.Xmat.T, self.Ymat))

    def predict(self, Xnew):
        if self.compute_spline:
            for x in Xnew:
                x.xbasis = self.spline_basis
                x.compute_spline_expansions()

        Xmat = self.get_spline_mat(Xnew)
        Ypred = np.zeros_like(Xmat)

        if self.fit_intercept:
            Xmat = np.hstack(
                [np.ones(Xmat.shape[0]).reshape(-1, 1), Xmat])

        out = []
        for i in range(Xmat.shape[0]):
            y_ = np.matmul(Xmat[i, :], self.beta)
            curr = Distribution(smooth_sigma=Xnew[0].smooth_sigma)
            curr.init_from_clr(
                self.spline_basis.xgrid, self.spline_basis.eval_spline(y_))
            out.append(curr)

        return out

    def get_spline_mat(self, distribs):
        """Stacks all the coefficient of the spline expansions by row
        """
        out = np.zeros((len(distribs), self.nbasis))
        for i, d in enumerate(distribs):
            out[i, :] = d.clr_coeffs

        eps = np.ones(out.shape[1]) * 1e-6
        for i in range(out.shape[1]):
            out[:, i] += np.sum(eps[:i])

        return out
