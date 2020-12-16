import numpy as np
from sklearn.base import BaseEstimator
from qpsolvers import solve_qp

from wtda.spline import MonotoneQuadraticSplineBasis
from wtda.wasserstein.distributions import Cdf, InvCdf


class ScalarOnFunctionRegression(BaseEstimator):
    """Scalar on functional regression.

    In the simplest form, for data \{(y_i, x_i)\}, where y_i \in R
    and x_i are functional covariates, the linear model minimizes the
    following loss function:

        LOSS = 0.5 \sum_{i=1}^n (y_i - <x_i, \beta>)^2

    and returns the optimal \beta.
    The inner product <f, g> is defined starting from the representation
    operator \phi(f): f \mapsto (\varphi_f,  F^{-}_f), where
    \varphi_f \in R^2 is a vector representing the area and the minimum
    value of the function, and F^{-}_f is instead the quantile function
    associated to the probability density function obtained by
    standardizing f.

         <f, g> = \lambda <\varphi_f, \varphi_g>_{\R^2} + 
            (1-\lambda) \int_0^1 F^{-}_f(x)  F^{-}_g(x) dx 


    In case for each response there is more than one functional covariate,
    the loss becomes

        LOSS = 0.5 \sum_{i=1}^n (y_i - \sum_{j=1}^q <x_{ij}, \beta_j>)^2

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

    def __init__(self, fit_intercept=True, nbasis=-1, spline_basis=None, 
                 compute_invcdf=True):
        self.fit_intercept = fit_intercept
        self.nbasis = nbasis
        self.spline_basis = spline_basis
        self.compute_invcdf = compute_invcdf


    def _initialize(self):
        if self.spline_basis is None:
            self.spline_basis = MonotoneQuadraticSplineBasis(
                self.nbasis, np.linspace(0, 1, 100))
        else:
            self.nbasis = self.spline_basis.nbasis

        self.spline_basis.eval_metric()

    
    def fit(self, X, y):
        """Fit Scalar on functional linear model

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

        self._coeffs = np.linalg.solve(
            np.dot(self.design_matrix.T, self.design_matrix), 
            np.dot(self.design_matrix.T, self.y))
        
        if self.fit_intercept:
            self.intercept = self._coeffs[0]
            self._coeffs = self._coeffs[1:]
        else:
            self.intercept = 0.0

        self.phi = self._coeffs[:2 * self.n_predictors].reshape(
            self.n_predictors, 2)
        self._beta = self._coeffs[2 * self.n_predictors:].reshape(
            self.n_predictors, self.nbasis)
        self._project()

    def predict(self, Xnew):
        """Predict a new value for covaraites Xnew

        Parameters
        ----------
        Xnew : array-like of shape (n_samples, n_predictors)
        """
        if self.compute_invcdf:
            for _, x in np.ndenumerate(Xnew):
                x.compute_inv_cdf(self.spline_basis)

        out = np.ones(Xnew.shape[0]) * self.intercept
        for i in range(Xnew.shape[0]):
            for j in range(self.n_predictors):
                out[i] += \
                    np.dot(Xnew[i, j].phi, self.phi[j, :]) + \
                    np.dot(Xnew[i, j].inv_cdf.inv_cdf_coeffs,
                           np.dot(self.spline_basis.metric, self.beta[j, :]))
        return out

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
        self.beta = self._beta

        # for j in range(self.n_predictors):
        #     beta_j = solve_qp(
        #         self.spline_basis.metric * 2,
        #         -2 * np.dot(self.spline_basis.metric, self._beta[j, :]),
        #         self.constraints_mat, self.cons_rhs)
        #     self.beta[j, :] = beta_j
