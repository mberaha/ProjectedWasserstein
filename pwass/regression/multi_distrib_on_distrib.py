import numpy as np
from sklearn.base import BaseEstimator
from qpsolvers import solve_qp
from scipy.integrate import trapz


from pwass.spline import MonotoneQuadraticSplineBasis
from pwass.distributions import Distribution


class MultiDistribOnDistribReg(BaseEstimator):
    def __init__(self, fit_intercept=True,  nbasis=-1, spline_basis=None,
                 compute_spline=True, lambda_ridge=0):
        self.fit_intercept = fit_intercept
        self.nbasis = nbasis
        self.spline_basis = spline_basis
        self.compute_spline = compute_spline
        self.lambda_ridge = lambda_ridge

    def _initialize(self):
        if self.spline_basis is None:
            self.spline_basis = MonotoneQuadraticSplineBasis(
                self.nbasis, np.linspace(0, 1, 100))
        else:
            self.nbasis = self.spline_basis.nbasis

        self.spline_basis.eval_metric()

        self.constraints_mat = np.zeros((self.nbasis - 1, self.nbasis))
        for i in range(self.nbasis-1):
            self.constraints_mat[i, i] = 1
            self.constraints_mat[i, i+1] = -1

        self.cons_rhs = np.zeros((self.nbasis-1,))

    def fit(self, X, Y):
        """
        X: np.array of shape (nobs, n_predictors)
        """
        self._initialize()
        self.n_samples = X.shape[0]
        self.n_preds = X.shape[1]
        self.X = X
        self.Y = Y

        if self.compute_spline:
            for x in np.nditer(self.X):
                x.wbasis = self.spline_basis
                x.compute_spline_expansions()

            for y in self.Y:
                y.wbasis = self.spline_basis
                y.compute_spline_expansions()

        self.Xmat = np.zeros((self.n_samples, self.nbasis * self.n_preds))
        for i in range(self.X.shape[0]):
            self.Xmat[i, :] = np.concatenate(
                [x.quantile_coeffs for x in self.X[i, :]])
        if self.fit_intercept:
            self.Xmat = np.hstack(
                [np.ones(self.n_samples).reshape(-1, 1), self.Xmat])

        self.Ymat = np.zeros((self.n_samples, self.nbasis))
        for i, y in enumerate(self.Y):
            self.Ymat[i, :] = y.quantile_coeffs

        XXtrans = np.matmul(self.Xmat.T, self.Xmat)
        self.beta = np.linalg.solve(
            XXtrans + self.lambda_ridge *
            np.eye(XXtrans.shape[0]),
            np.matmul(self.Xmat.T, self.Ymat))

    def predict(self, Xnew):
        if self.compute_spline:
            for x in np.nditer(Xnew):
                x.wbasis = self.spline_basis
                x.compute_spline_expansions()

        Ypred = np.zeros((Xnew.shape[0], self.nbasis))

        Xmat = np.zeros((Xnew.shape[0], self.nbasis * self.n_preds))
        for i in range(Xnew.shape[0]):
            Xmat[i, :] = np.concatenate(
                [x.quantile_coeffs for x in Xnew[i, :]])
        if self.fit_intercept:
            Xmat = np.hstack(
                [np.ones(Xmat.shape[0]).reshape(-1, 1), Xmat])

        out = []
        for i in range(Xmat.shape[0]):
            y_ = np.matmul(Xmat[i, :], self.beta)
            # Ypred[i, :] = self.project(y_)
            Ypred[i, :] = y_
            curr = Distribution(
                wbasis=self.spline_basis,
                smooth_sigma=Xnew[0, 0].smooth_sigma)
            curr.init_from_quantile(
                self.spline_basis.xgrid, self.spline_basis.eval_spline(
                    Ypred[i, :]))
            curr.quantile_coeffs = Ypred[i, :]
            out.append(curr)

        return out

    def project(self, y):
        return solve_qp(
            self.spline_basis.metric * 2,
            -2 * np.dot(self.spline_basis.metric, y),
            self.constraints_mat, self.cons_rhs)

    def score(self, X, ytrue, return_sd=False):
        ypred = self.predict(X)
        errs = np.zeros(len(ypred))
        out = 0.0
        for i in range(len(ypred)):
            ytrue_eval = self.spline_basis.eval_spline(
                ytrue[i].quantile_coeffs)
            ypred_eval = self.spline_basis.eval_spline(
                ypred[i].quantile_coeffs)

            errs[i] = trapz((ytrue_eval - ypred_eval)**2,
                         self.spline_basis.xgrid)
        if return_sd:
            return np.mean(errs), np.std(errs)
        else:
            return -np.mean(errs)
