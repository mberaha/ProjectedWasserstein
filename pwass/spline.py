import logging
import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import splev, splrep
from scipy.optimize import least_squares, minimize
from qpsolvers import solve_qp


class SplineBasis(object):
    def __init__(self, deg, knots=None, xgrid=None, nbasis=None):
        self.deg = deg
        if knots is not None:
            self.knots = knots
            self.xgrid = np.linspace(knots[0], knots[-1], 200)

        elif xgrid is not None and nbasis is not None:
            self.xgrid = xgrid
            self.nbasis = nbasis
            x_ = np.linspace(xgrid[0], xgrid[-1], nbasis)
            y_ = np.random.normal(size=nbasis)
            self.knots, _, deg = splrep(x_, y_, k=deg)

        self.B = np.zeros((self.nbasis, len(self.xgrid)))
        for i in range(self.nbasis):
            c_ = np.zeros(self.nbasis)
            c_[i] = 1
            self.B[i, :] = self.eval_spline(c_)

        self.BBtrans = np.matmul(self.B, self.B.T)
        
        self.eval_metric()

    def eval_metric(self):
        delta = self.xgrid[1] - self.xgrid[0]
        self.spline_basis = np.zeros((self.nbasis, len(self.xgrid)))
        for i in range(self.nbasis):
            coeffs = np.zeros(self.nbasis)
            coeffs[i] = 1.0
            self.spline_basis[i, :] = splev(
                self.xgrid, (self.knots, coeffs, self.deg))

        self.metric = np.zeros((self.nbasis, self.nbasis))
        for i in range(self.nbasis):
            for j in range(self.nbasis):
                self.metric[i, j] = np.sum(
                    self.spline_basis[i, :] * self.spline_basis[j, :]) * delta

    def get_spline_expansion(self, f_eval, xgrid=None):
        """
        Returns the coefficient of the spline expansion for f_eval
        """
        def _cost(f, coeffs):
            return np.sum((
                f - splev(xgrid, (self.knots, coeffs, self.deg))) ** 2)

        if xgrid is None or np.all(xgrid == self.xgrid):
            B = self.B
            BBtrans = self.BBtrans
        else:
            B = np.zeros((self.nbasis, len(xgrid)))
            for i in range(self.nbasis):
                c_ = np.zeros(self.nbasis)
                c_[i] = 1
                B[i, :] = self.eval_spline(c_, xgrid)
            BBtrans = np.matmul(B, B.T)
        
        # try:
        #     out = np.linalg.solve(
        #         BBtrans, np.dot(B, f_eval.reshape(-1, 1))).reshape(-1,)
        # except Exception as e:
        #     out = np.linalg.lstsq(
        #         BBtrans, np.dot(B, f_eval.reshape(-1, 1))).reshape(-1,)
        out = lstsq(BBtrans, np.dot(B, f_eval.reshape(-1, 1)),
                    lapack_driver="gelsy")[0].reshape(-1,)
        return out

    def eval_spline(self, coeffs, xgrid=None):
        if xgrid is None:
            xgrid = self.xgrid

        return splev(xgrid, (self.knots, coeffs, self.deg))

    def eval_spline_der(self, coeffs, xgrid=None):
        if xgrid is None:
            xgrid = self.xgrid

        return splev(xgrid, (self.knots, coeffs, self.deg), der=1)


class MonotoneQuadraticSplineBasis(SplineBasis):
    def __init__(self, nbasis, xgrid):
        super().__init__(2, xgrid=xgrid, nbasis=nbasis)
        self.constraints_mat = np.zeros((self.nbasis - 1, self.nbasis))
        for i in range(self.nbasis-1):
            self.constraints_mat[i, i] = 1
            self.constraints_mat[i, i+1] = -1

        self.cons_rhs = np.zeros((self.nbasis-1,))
        

    def get_spline_expansion(self, f_eval, xgrid=None):
        """
        Returns the coefficient of the spline expansion for f_eval
        """
        out = super().get_spline_expansion(f_eval, xgrid)
        proj = solve_qp(
            self.metric * 2,
            -2 * np.dot(self.metric, out),
            self.constraints_mat, self.cons_rhs)

        return  proj
