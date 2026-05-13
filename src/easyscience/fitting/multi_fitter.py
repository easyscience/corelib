# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

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
        fit_objects: Optional[List] = None,
        fit_functions: Optional[List[Callable]] = None,
    ):
        # Create a dummy core object to hold all the fit objects.
        self._fit_objects = CollectionBase('multi', *fit_objects)
        self._fit_functions = fit_functions
        # Initialize with the first of the fit_functions, without this it is
        # not possible to change the fitting engine.
        super().__init__(self._fit_objects, self._fit_functions[0])

    def _fit_function_wrapper(
        self, real_x: Optional[List[np.ndarray]] = None, flatten: bool = True
    ) -> Callable:
        """
        Simple fit function which injects the N real X (independent)
        values into the optimizer function.

        This will also flatten the results if needed.

        Parameters
        ----------
        real_x : Optional[List[np.ndarray]], default=None
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
        x: List[np.ndarray],
        y: List[np.ndarray],
        weights: Optional[List[np.ndarray]],
        vectorized: bool,
    ) -> tuple[
        np.ndarray, List[np.ndarray], np.ndarray, Optional[np.ndarray], List[tuple[int, ...]]
    ]:
        """
        Convert an array of X's and Y's  to an acceptable shape for
        fitting.

        Parameters
        ----------
        x : List[np.ndarray]
            List of independent variables.
        y : List[np.ndarray]
            List of dependent variables.
        weights : Optional[List[np.ndarray]]
            Optional weights for each dataset.
        vectorized : bool
            Is the fn input vectorized or point based?

        Returns
        -------
        tuple[np.ndarray, List[np.ndarray], np.ndarray, Optional[np.ndarray], List[tuple[int, ...]]]
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
        x: List[np.ndarray],
        y: List[np.ndarray],
    ) -> List[FitResults]:
        """
        Split a multi-fit result object back into per-dataset results.

        Parameters
        ----------
        fit_result_obj : FitResults
            Combined fit result returned by the minimizer.
        x : List[np.ndarray]
            Original x coordinates for each dataset.
        y : List[np.ndarray]
            Original y coordinates for each dataset.

        Returns
        -------
        List[FitResults]
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

    def sample(
        self,
        x: List[np.ndarray],
        y: List[np.ndarray],
        weights: List[np.ndarray],
        samples: int = 10000,
        burn: int = 1000,
        thin: int = 10,
        chains: int | None = None,
        population: int | None = None,
        seed: int | None = None,
        vectorized: bool = False,
        sampler_kwargs: dict | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        abort_test: Callable[[], bool] | None = None,
    ) -> Dict:
        """Run Bayesian MCMC sampling using the BUMPS DREAM sampler.

        Requires that the current minimizer is a BUMPS instance (i.e. the
        minimizer was switched to ``AvailableMinimizers.Bumps`` or equivalent).

        Parameters
        ----------
        x : List[np.ndarray]
            List of independent variable arrays (one per dataset).
        y : List[np.ndarray]
            List of dependent variable arrays (one per dataset).
        weights : List[np.ndarray]
            List of weight arrays (one per dataset).
        samples : int, default=10000
            Number of retained DREAM samples requested from BUMPS.
        burn : int, default=1000
            Burn-in steps.
        thin : int, default=10
            Thinning interval.
        chains : int | None, default=None
            User-friendly alias for BUMPS DREAM population count.
        population : int | None, default=None
            BUMPS DREAM population count (``pop``) for advanced users.
        seed : int | None, default=None
            Best-effort random seed. BUMPS DREAM may use additional internal
            RNG state that is not controlled by this seed, so exact
            reproducibility is not guaranteed.
        vectorized : bool, default=False
            Whether the fit function expects vectorized (multidimensional)
            input.
        sampler_kwargs : dict | None, default=None
            Additional keyword arguments forwarded to the BUMPS DREAM sampler
            via `bumps.fitters.fit`.
        progress_callback : Callable[[dict], bool | None] | None, default=None
            Optional callback for progress updates during sampling.  The
            payload dict includes ``iteration`` (DREAM generation number) and
            ``sampling: True``.
        abort_test : Callable[[], bool] | None, default=None
            Optional callback that returns ``True`` to signal that sampling
            should be aborted. Called periodically during the DREAM sampling
            loop.

        Returns
        -------
        Dict
            Dictionary with keys ``'draws'``, ``'param_names'``, ``'state'``,
            and ``'logp'``.

        Raises
        ------
        RuntimeError
            If the current minimizer is not a BUMPS instance.
        ValueError
            If both ``chains`` and ``population`` are provided with different
            values.
        """
        # --- Alias resolution ---
        if chains is not None and population is not None:
            if chains != population:
                raise ValueError(
                    f'Conflicting population arguments: chains={chains}, population={population}. '
                    'Only provide one.'
                )
            pop = chains
        elif chains is not None:
            pop = chains
        elif population is not None:
            pop = population
        else:
            pop = None

        # Flatten multi-dataset arrays
        x_fit, x_new, y_new, w_new, _dims = self._precompute_reshaping(
            x, y, weights, vectorized=vectorized
        )
        self._dependent_dims = _dims

        # Wrap fit functions for multi-dataset flattening, mirroring the
        # ``Fitter.fit`` lifecycle: use the property setter so the minimizer
        # is re-created with the wrapped fit function.
        original_fit_func = self.fit_function
        fit_fun_wrap = self._fit_function_wrapper(x_new, flatten=True)
        self.fit_function = fit_fun_wrap

        try:
            minimizer = self.minimizer

            # Verify it's a BUMPS minimizer (sampling only works with BUMPS/DREAM)
            if not (hasattr(minimizer, 'package') and minimizer.package == 'bumps'):
                raise RuntimeError(
                    'Bayesian sampling requires a BUMPS minimizer. '
                    'Use ``fitter.switch_minimizer(AvailableMinimizers.Bumps)`` first.'
                )

            # Delegate to the BUMPS minimizer's public sample method
            result = minimizer.sample(
                x=x_fit,
                y=y_new,
                weights=w_new,
                samples=samples,
                burn=burn,
                thin=thin,
                chains=None,  # alias already resolved into `pop`
                population=pop,
                seed=seed,
                sampler_kwargs=sampler_kwargs,
                progress_callback=progress_callback,
                abort_test=abort_test,
            )
        finally:
            self.fit_function = original_fit_func

        return result
