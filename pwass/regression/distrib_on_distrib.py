import numpy as np
from sklearn.base import BaseEstimator
from qpsolvers import solve_qp
from scipy.integrate import trapz

from pwass.spline import MonotoneQuadraticSplineBasis
from pwass.distributions import Distribution


class DistribOnDistribReg(BaseEstimator):
    def __init__(self, fit_intercept=True, nbasis=-1, spline_basis=None,
                 compute_spline=True, lambda_ridge=0, rho_ps=0.0,
                 method="ridge"):
        self.fit_intercept = fit_intercept
        self.nbasis = nbasis
        self.spline_basis = spline_basis
        self.compute_spline = compute_spline
        self.lambda_ridge = lambda_ridge
        self.rho_ps = rho_ps
        self.method = method

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
        self._initialize()
        self.n_samples = len(X)
        self.X = X
        self.Y = Y

        if self.compute_spline:
            for x in self.X:
                x.wbasis = self.spline_basis
                x.compute_spline_expansions()

            for y in self.Y:
                y.wbasis = self.spline_basis
                y.compute_spline_expansions()

        self.Xmat = self.get_spline_mat(self.X)
        self.Ymat = self.get_spline_mat(self.Y)
        if self.fit_intercept:
            self.Xmat = np.hstack(
                [np.ones(self.n_samples).reshape(-1, 1), self.Xmat])
        if self.method == "ridge":
            self._fit_ridge()
        else:
            assert self.fit_intercept == False
            self._fit_ps()

    def predict(self, Xnew):
        if self.compute_spline:
            for x in Xnew:
                x.wbasis = self.spline_basis
                x.compute_spline_expansions()

        Xmat = self.get_spline_mat(Xnew)
        Ypred = np.zeros_like(Xmat)

        if self.fit_intercept:
            Xmat = np.hstack(
                [np.ones(Xmat.shape[0]).reshape(-1, 1), Xmat])

        out = []
        for i in range(Xmat.shape[0]):
            if self.method == "ridge":
                y_ = np.matmul(Xmat[i, :], self.beta)
            else:
                y_ = np.matmul(
                    self.beta, np.matmul(self.spline_basis.metric, X[i, :]))
            Ypred[i, :] = self.project(y_)
            curr = Distribution(
                wbasis=self.spline_basis,
                smooth_sigma=Xnew[0].smooth_sigma)
            curr.init_from_quantile(
                self.spline_basis.xgrid, self.spline_basis.eval_spline(
                    Ypred[i, :]))
            curr.quantile_coeffs = Ypred[i, :]
            out.append(curr)

        return out

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

    def project(self, y):
        out =  solve_qp(
            self.spline_basis.metric * 2,
            -2 * np.dot(self.spline_basis.metric, y),
            self.constraints_mat, self.cons_rhs)
        return out

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

    def _fit_ridge(self):
        XXtrans = np.matmul(self.Xmat.T, self.Xmat)
        self.beta = np.linalg.solve(
            XXtrans + self.lambda_ridge * np.eye(XXtrans.shape[0]),
            np.matmul(self.Xmat.T, self.Ymat))

    def _fit_ps(self):
        self._compute_e_prime()
        Chat = np.zeros((self.nbasis, self.nbasis))
        Dhat = np.zeros((self.nbasis, self.nbasis))
        E = self.spline_basis.metric
        for k in range(self.nbasis):
            ek = np.zeros(self.nbasis)
            ek[k] = 1.0
            inner_prods = np.matmul(self.Xmat, np.matmul(
                self.spline_basis.metric, ek))
            X_times_inner_prods = self.Xmat * inner_prods[:, np.newaxis]
            Y_times_inner_prods = self.Ymat * inner_prods[:, np.newaxis]
            for s in range(self.nbasis):
                es = np.zeros(self.nbasis)
                es[s] = 1.0
                Chat[k, s] = np.mean(
                    np.matmul(X_times_inner_prods, np.matmul(E, es)))
                Dhat[k, s] = np.mean(
                    np.matmul(Y_times_inner_prods, np.matmul(E, es)))

        rho = 1e-8
        Crho = np.kron(E, Chat + rho * self.Eprime)
        P = np.kron(self.Eprime, self.spline_basis.metric) + \
                np.kron(self.spline_basis.metric, self.Eprime)
        vecbeta_hat = np.linalg.solve(Crho + rho * P, Dhat.T.reshape(-1, 1))
        self.beta = vecbeta_hat.reshape(self.nbasis, self.nbasis)

    def _compute_e_prime(self):
        self.Eprime = np.zeros_like(self.spline_basis.metric)
        for i in range(self.nbasis):
            ci = np.zeros(self.nbasis)
            ci[i] = 1
            curr = self.spline_basis.eval_spline_der(ci)
            self.Eprime[i, i] = trapz(curr * curr, self.spline_basis.xgrid)
            for j in range(i):
                cj = np.zeros(self.nbasis)
                cj[j] = 1
                other = self.spline_basis.eval_spline_der(cj)
                self.Eprime[i, j] = trapz(curr * other, self.spline_basis.xgrid)
                self.Eprime[j, i] = self.Eprime[i, j]
