# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np


class FitResults:
    """At the moment this is just a dummy way of unifying the returned
    fit parameters.
    """

    __slots__ = [
        'success',
        'minimizer_engine',
        'fit_args',
        'p',
        'p0',
        'x',
        'x_matrices',
        'y_obs',
        'y_calc',
        'y_err',
        'n_evaluations',
        'iterations',
        'message',
        'engine_result',
        'total_results',
    ]

    def __init__(self):
        """Init function."""
        self.success = False
        self.minimizer_engine = None
        self.fit_args = {}
        self.p = {}
        self.p0 = {}
        self.x = np.ndarray([])
        self.x_matrices = np.ndarray([])
        self.y_obs = np.ndarray([])
        self.y_calc = np.ndarray([])
        self.y_err = np.ndarray([])
        self.n_evaluations = None
        self.iterations = None
        self.message = ''
        self.engine_result = None
        self.total_results = None

    def __repr__(self) -> str:
        """Repr function."""
        engine_name = self.minimizer_engine.__name__ if self.minimizer_engine else None
        try:
            chi2_val = self.chi2
            reduced_val = self.reduced_chi2
            if not np.isfinite(chi2_val) or not np.isfinite(reduced_val):
                raise ValueError('Chi2 or reduced chi2 is not finite')
            chi2 = f'{chi2_val:.4g}'
            reduced = f'{reduced_val:.4g}'
        except Exception:
            chi2 = 'N/A'
            reduced = 'N/A'

        try:
            n_points = len(self.x)
        except TypeError:
            n_points = 0

        lines = [
            f'FitResults(success={self.success}',
            f'  n_pars={self.n_pars}, n_points={n_points}',
            f'  chi2={chi2}, reduced_chi2={reduced}',
            f'  n_evaluations={self.n_evaluations}',
            f'  iterations={self.iterations}',
            f'  minimizer={engine_name}',
        ]
        if self.message:
            lines.append(f"  message='{self.message}'")
        if self.p:
            par_str = ', '.join(f'{k}={v:.4g}' for k, v in self.p.items())
            lines.append(f'  parameters={{{par_str}}}')
        lines.append(')')
        return '\n'.join(lines)

    @property
    def n_pars(self):
        """N pars."""
        return len(self.p)

    @property
    def residual(self):
        """Residual function."""
        return self.y_obs - self.y_calc

    @property
    def chi2(self):
        """Chi2 function."""
        return ((self.residual / self.y_err) ** 2).sum()

    @property
    def reduced_chi2(self):
        """Reduced chi2."""
        return self.chi2 / (len(self.x) - self.n_pars)


class FitError(Exception):
    def __init__(self, e: Exception = None):
        """Init function."""
        self.e = e

    def __str__(self) -> str:
        """Str function."""
        s = ''
        if self.e is not None:
            s = f'{self.e}\n'
        return s + 'Something has gone wrong with the fit'
