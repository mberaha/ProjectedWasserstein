import logging
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator, splev, splrep
from scipy.integrate import cumtrapz, simps
from scipy.ndimage import gaussian_filter1d

from pwass.spline import MonotoneQuadraticSplineBasis



class Distribution(object):
    """
    General class to represent a distribution
    """
    def __init__(
            self, xbasis=None, wbasis=None, smooth_sigma=1.0):
        self.thr = 1e-8
        self.smooth_sigma = smooth_sigma
        self.xbasis = xbasis
        self.wbasis = wbasis
        self.clr_eval = None

    def init_from_pdf(self, pdf_grid, pdf_eval):
        self.pdf_grid = pdf_grid
        self.pdf_eval = pdf_eval

        self.cdf_grid = pdf_grid
        self.cdf_eval = cumtrapz(pdf_eval, self.pdf_grid, initial=0)

        self._invert_cdf()

    def init_from_cdf(self, cdf_grid, cdf_eval):
        self.cdf_grid = cdf_grid
        self.cdf_eval = cdf_eval
        self._invert_cdf()

        self.pdf_eval = (cdf_eval[1:] - cdf_eval[:-1]) / np.diff(self.cdf_grid)
        self.pdf_grid = self.cdf_grid[1:]

        if self.xbasis is not None:
            self.pdf_coeffs = self.xbasis.get_spline_expansion(
                self.pdf_eval, self.pdf_grid)

            self.cdf_coeffs = self.xbasis.get_spline_expansion(
                self.cdf_eval, self.cdf_grid)

    def init_from_quantile(self, quantile_grid, quantile_eval):
        self.quantile_grid = quantile_grid
        self.quantile_eval = quantile_eval
    
    def _invert_cdf(self):
        keep = np.where(np.diff(self.cdf_eval) > 1e-5)
        self.quantile_grid = self.cdf_eval[keep]
        self.quantile_eval = self.cdf_grid[keep]

    def _invert_quantile(self):
        cdf = PchipInterpolator(self.quantile_eval, self.quantile_grid)
        self.cdf_grid = np.linspace(
            self.quantile_eval[0], self.quantile_eval[-1], 1000)
        self.pdf_grid = self.cdf_grid
        
        pdf_eval = gaussian_filter1d(
            cdf.derivative()(self.pdf_grid), sigma=self.smooth_sigma)
        self.pdf_eval = pdf_eval / simps(pdf_eval, self.cdf_grid)
        self.cdf_eval = cumtrapz(self.pdf_eval, self.pdf_grid, initial=0)

    def compute_spline_expansions(self):
        if self.xbasis is not None:
            if self.clr_eval is not None:
                self.clr_coeffs = self.xbasis.get_spline_expansion(
                    self.clr_eval, self.clr_grid)

        if self.wbasis is not None:
            try:
                self.quantile_coeffs = self.wbasis.get_spline_expansion(
                    self.quantile_eval, self.quantile_grid)
            except Exception as e:
                from scipy.interpolate import PchipInterpolator
                interp = PchipInterpolator(self.quantile_grid, self.quantile_eval)
                self.quantile_grid = np.linspace(0, 1, 200)
                self.quantile_eval = interp(self.quantile_grid)
                self.quantile_coeffs = self.wbasis.get_spline_expansion(
                    self.quantile_eval, self.quantile_grid)

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

    def compute_clr(self):
        self.clr_grid = self.pdf_grid
        self.clr_eval = self.clr(self.pdf_eval, self.pdf_grid)
        if self.xbasis is not None:
            self.clr_coeffs = self.xbasis.get_spline_expansion(
                self.clr_eval, self.clr_grid)

    def init_from_clr(self, clr_grid, clr_eval):
        self.clr_grid = clr_grid
        self.clr_eval = clr_eval
        self.init_from_pdf(clr_grid, self.inv_clr(clr_eval, clr_grid))




