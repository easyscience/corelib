# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Callable

import numpy as np

from ..base_classes import CollectionBase
from .fitter import Fitter
from .minimizers import FitResults


class MultiFitter(Fitter):
    """
    Extension of Fitter to enable multiple dataset/fit function fitting.

    We can fit these types of data simultaneously:
    - Multiple models on multiple datasets.

    The inherited ``fit`` wrapper from ``Fitter`` is used unchanged,
    including support for forwarding progress callbacks to the active
    minimizer.
    """

    def __init__(
        self,
        fit_objects: list | None = None,
        fit_functions: list[Callable] | None = None,
    ):
        # Create a dummy core object to hold all the fit objects.
        self._fit_objects = CollectionBase('multi', *fit_objects)
        self._fit_functions = fit_functions
        # Initialize with the first of the fit_functions, without this it is
        # not possible to change the fitting engine.
        super().__init__(self._fit_objects, self._fit_functions[0])

    def _fit_function_wrapper(
        self, real_x: list[np.ndarray] | None = None, flatten: bool = True
    ) -> Callable:
        """
        Simple fit function which injects the N real X (independent)
        values into the optimizer function.

        This will also flatten the results if needed.

        Parameters
        ----------
        real_x : list[np.ndarray] | None, default=None
            List of independent x parameters to be injected. By default,
            None.
        flatten : bool, default=True
            Should the result be a flat 1D array? By default, True.

        Returns
        -------
        Callable
            Wrapped optimizer function.
        """
        # Extract of a list of callable functions
        wrapped_fns = []
        for this_x, this_fun in zip(real_x, self._fit_functions):
            self._fit_function = this_fun
            wrapped_fns.append(Fitter._fit_function_wrapper(self, this_x, flatten=flatten))

        def wrapped_fun(x, **kwargs):
            # Generate an empty Y based on x
            y = np.zeros_like(x)
            i = 0
            # Iterate through wrapped functions, passing the WRONG x, the correct
            # x was injected in the step above.
            for idx, dim in enumerate(self._dependent_dims):
                ep = i + np.prod(dim)
                y[i:ep] = wrapped_fns[idx](x, **kwargs)
                i = ep
            return y

        return wrapped_fun

    @staticmethod
    def _precompute_reshaping(
        x: list[np.ndarray],
        y: list[np.ndarray],
        weights: list[np.ndarray] | None,
        vectorized: bool,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray | None, list[tuple[int, ...]]]:
        """
        Convert an array of X's and Y's  to an acceptable shape for
        fitting.

        Parameters
        ----------
        x : list[np.ndarray]
            List of independent variables.
        y : list[np.ndarray]
            List of dependent variables.
        weights : list[np.ndarray] | None
            Optional weights for each dataset.
        vectorized : bool
            When ``True``, each x array may be multi-dimensional (e.g.
            an ``(N, M, 2)`` grid for a 2D model) and is left as-is.
            When ``False`` (default), each x array is expected to be
            1-D.

        Returns
        -------
        tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray | None, list[tuple[int, ...]]]
            Reshaped x values, reshaped input data, flattened y values,
            flattened weights, and stored dependent dimensions.
        """
        if weights is None:
            weights = [None] * len(x)
        _, _x_new, _y_new, _weights, _dims = Fitter._precompute_reshaping(
            x[0], y[0], weights[0], vectorized
        )
        x_new = [_x_new]
        y_new = [_y_new]
        w_new = [_weights]
        dims = [_dims]
        for _x, _y, _w in zip(x[1::], y[1::], weights[1::]):
            _, _x_new, _y_new, _weights, _dims = Fitter._precompute_reshaping(
                _x, _y, _w, vectorized
            )
            x_new.append(_x_new)
            y_new.append(_y_new)
            w_new.append(_weights)
            dims.append(_dims)
        y_new = np.hstack(y_new)
        if w_new[0] is None:
            w_new = None
        else:
            w_new = np.hstack(w_new)
        x_fit = np.linspace(0, y_new.size - 1, y_new.size)
        return x_fit, x_new, y_new, w_new, dims

    def _post_compute_reshaping(
        self,
        fit_result_obj: FitResults,
        x: list[np.ndarray],
        y: list[np.ndarray],
    ) -> list[FitResults]:
        """
        Split a multi-fit result object back into per-dataset results.

        Parameters
        ----------
        fit_result_obj : FitResults
            Combined fit result returned by the minimizer.
        x : list[np.ndarray]
            Original x coordinates for each dataset.
        y : list[np.ndarray]
            Original y coordinates for each dataset.

        Returns
        -------
        list[FitResults]
            One fit result object per dataset.
        """

        cls = fit_result_obj.__class__
        sp = 0
        fit_results_list = []
        for idx, this_x in enumerate(x):
            # Create a new Results obj
            current_results = cls()
            ep = sp + int(np.array(self._dependent_dims[idx]).prod())

            #  Fill out the new result obj (see EasyScience.fitting.Fitting_template.FitResults)
            current_results.success = fit_result_obj.success
            current_results.minimizer_engine = fit_result_obj.minimizer_engine
            current_results.p = fit_result_obj.p
            current_results.p0 = fit_result_obj.p0
            current_results.n_evaluations = fit_result_obj.n_evaluations
            current_results.iterations = fit_result_obj.iterations
            current_results.message = fit_result_obj.message
            current_results.x = this_x
            current_results.y_obs = y[idx]
            current_results.y_calc = np.reshape(
                fit_result_obj.y_calc[sp:ep], current_results.y_obs.shape
            )
            current_results.y_err = np.reshape(
                fit_result_obj.y_err[sp:ep], current_results.y_obs.shape
            )
            current_results.engine_result = fit_result_obj.engine_result

            # Attach an additional field for the un-modified results
            current_results.total_results = fit_result_obj
            fit_results_list.append(current_results)
            sp = ep
        return fit_results_list
