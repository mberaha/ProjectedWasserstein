import numpy as np
import sklearn.metrics as metrics
from scipy.interpolate import splev
from scipy.interpolate import splrep
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from qpsolvers import solve_qp


from wtda.spline import MonotoneQuadraticSplineBasis
from wtda.wasserstein.distributions import Cdf, InvCdf


def _nll(astar, y, design_matrix, nreg, nbasis, pen1, pen2=0):
    sigmoids = 1.0 / (1.0 + np.exp(- np.dot(design_matrix, astar)))
    nll = -np.sum(y * np.log(sigmoids + 1e-5)) - \
        np.sum((1 - y) * np.log(1 - sigmoids + 1e-5))
    aphi = astar[1 : (1 + nreg * 2)]
    abeta = astar[1 + nreg * 2: ]
    return (nll + pen1 * np.sum(abeta ** 2) + pen2 * np.sum(aphi ** 2) +
            pen1 * astar[0] ** 2) 


def _grad_fn(astar, y, design_matrix, nreg, nbasis, pen1, pen2=0):
    sigmoids = 1.0 / (1.0 + np.exp(- np.dot(design_matrix, astar)))
    err = sigmoids - y
    grad = np.zeros_like(astar)

    grad = np.sum(
        design_matrix * err.reshape(-1, 1), 0)

    grad[0] += pen1 * astar[0]
    grad[1: (1 + nreg * 2)] += pen2 * astar[1: (1 + nreg * 2)]
    grad[1 + nreg * 2:] += pen1 * astar[1 + nreg * 2:]

    return grad


def _make_constraint(i):
    return lambda x: -x[i + 2] + x[i + 3]


class FunctionalLogisticRegression(BaseEstimator):
    """Functional logistic regression.

    For data \{(y_i, x_{i1}, ... x_{iq})\}, the logistic functional model is:
        logit(P(Y_i = 1)) = \beta_0 + \sum_{j=1}^q <x_{ij}, \beta_j>

    where logit(x) = ln(x / (1-x)).
    Inference is carried out by minimizing the negative log likelihood.

    Parameters
    ----------
    fit_intercept : bool, default True
        If True, an intercept term is added to the covariates

    nbasis : int, default -1 (meaning that the spline_basis param will be used)
        Number of quadratic splines to use as basis for the 
        space L^{2\arrowup}([0, 1]), i.e. the space of monothonically
        increasing functions that are also square integrable on [0, 1].
        An element in this space represents a quantile function

    spline_basis : instance of MonotoneQuadraticSplineBasis object, default None
        A MonotoneQuadraticSplineBasis for the interval [0, 1] 

    maxiter : int, default 100
        Maximum number of iterations for the iterative algorithm that
        minimzes the negative log likelihood

    toll : float, default 1e-3
        Tolerance of the iterative algorithm

    pen_beta:
        Ridge-regression penalty to be applied to the coefficients 
        \beta_0 and to the spline expansion coefficients of F^{-1}_\beta

    pen_phi:
        Rigde-regression penalty for the parametrs \varphi_\beta

    compute_invcdf: bool, default=True
        If True, the method compute_invcdf() will be called on each
        predictor.
        If this algorithm is used in a cross-validation, it is more
        efficient to call compute_inv_cdf() on the predictors before
        the cross_validation and pass compute_invcdf=False.
    """
    def __init__(
        self, fit_intercept=True, nbasis=-1, spline_basis=None,
        maxiter=100, toll=1e-3, pen_beta=1e-3, pen_phi=0,
        compute_invcdf=True):
        self.fit_intercept = fit_intercept
        self.nbasis = nbasis
        self.maxiter = maxiter
        self.toll = toll

        self.pen_beta = pen_beta
        self.pen_phi = pen_phi
        self.spline_basis = spline_basis
        self.compute_invcdf = compute_invcdf

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


    def fit(self, X, y):
        """Fit logistic functional linear model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_predictors)

        y: array-like of shape (nsamples, )
        """

        self._initialize()

        self.n_samples = X.shape[0]
        self.n_predictors = X.shape[1]
        self.X = X

        if self.compute_invcdf:
            for _, x in np.ndenumerate(self.X):
                x.compute_inv_cdf(self.spline_basis)
        self.y = y

        self.spline_coeffs = self._spline_expansion(self.X)
        self.design_matrix = self._design_matrix(self.X, self.spline_coeffs)

        x0_ = np.zeros(1 + 2 * self.n_predictors) if self.fit_intercept \
            else np.zeros(2 * self.n_predictors)
        
        curr = np.concatenate([x0_, np.zeros(self.n_predictors * self.nbasis)])
        prev_nll = _nll(curr, self.y, self.design_matrix, self.n_predictors,
                        self.nbasis, self.pen_beta, self.pen_phi)

        stepsize = 0.001
        # for i in range(self.maxiter):
        #     curr = curr - stepsize * _grad_fn(
        #         curr, self.y, self.design_matrix, self.n_predictors,
        #         self.nbasis, self.pen_beta, self.pen_phi)

        #     curr_nll = _nll(curr, self.y, self.design_matrix, self.n_predictors,
        #                     self.nbasis, self.pen_beta, self.pen_phi)

        #     if np.abs(curr_nll - prev_nll) > self.toll:
        #         break

        # self._coeff = curr
            
        self._coeff = minimize(
            _nll, method="BFGS",
            args=(self.y, self.design_matrix, self.n_predictors, 
                  self.nbasis, self.pen_beta, self.pen_phi),
            x0=np.concatenate(
                [x0_, np.zeros(self.n_predictors * self.nbasis)]),
            tol=self.toll,
            options={"maxiter": self.maxiter}).x

        if self.fit_intercept:
            self.intercept = self._coeff[0]
            self._coeff = self._coeff[1:]
        else:
            self.intercept = 0

        self.phi = self._coeff[:2 * self.n_predictors].reshape(
            self.n_predictors, 2)
        self._beta = self._coeff[2 * self.n_predictors:].reshape(
            self.n_predictors, self.nbasis)
        self._project()

    def predict(self, Xnew):
        """Predict a new value for covaraites Xnew

        Parameters
        ----------
        Xnew : array-like of shape (n_samples, n_predictors)
        """
        if self.compute_invcdf:
            for _, x in np.ndenumerate(self.X):
                x.compute_inv_cdf(self.spline_basis)

        scores = np.ones(Xnew.shape[0]) * self.intercept
        for i in range(Xnew.shape[0]):
            for j in range(self.n_predictors):
                scores[i] += \
                    np.dot(Xnew[i, j].phi, self.phi[j, :]) + \
                    np.dot(Xnew[i, j].inv_cdf.inv_cdf_coeffs,
                           np.dot(self.spline_basis.metric, self.beta[j, :]))
        
        preds = 1.0 / (1.0 + np.exp(-scores))
        preds[preds >= 0.5] = 1.0
        preds[preds < 0.5] = 0.0
        return preds

    def score(self, X, ytrue):
        """Score a prediction against the ground truth, using the 
        classification accuracy as metric

        Parameters
        ----------
        X : array-like of shape (n_samples, n_predictors)
        
        y: array-like of shape (nsamples, )
        """
        preds = self.predict(X)
        return metrics.accuracy_score(ytrue, preds)

    def _spline_expansion(self, X):
        spline_coeffs = np.zeros(
            (X.shape[0], self.n_predictors, self.nbasis))
        for i in range(X.shape[0]):
            for j in range(self.n_predictors):
                spline_coeffs[i, j, :] = X[i, j].inv_cdf.inv_cdf_coeffs
        return spline_coeffs

    def _design_matrix(self, X, spline_coeffs):
        design_matrix = np.zeros(
            (X.shape[0], (self.nbasis + 2) * self.n_predictors))
        design_matrix[:, :2 * self.n_predictors] = np.vstack(
            [np.concatenate([f.phi for f in X[i, :]])
             for i in range(X.shape[0])])
        design_matrix[:, 2 * self.n_predictors:] = np.vstack([
            np.matmul(self.spline_basis.metric,
                      spline_coeffs[i, :, :].T).T.reshape(-1,)
            for i in range(X.shape[0])])
        if self.fit_intercept:
            design_matrix = np.hstack(
                [np.ones((self.n_samples, 1)), design_matrix])
        return design_matrix

    def _project(self):
        # self.beta = np.zeros((self.n_predictors, self.nbasis))
        self.beta = self._beta
        # for j in range(self.n_predictors):
        #     beta_j = solve_qp(
        #         self.spline_basis.metric * 2,
        #         -2 * np.dot(self.spline_basis.metric, self._beta[j, :]),
        #         self.constraints_mat, self.cons_rhs)
        #     self.beta[j, :] = beta_j
