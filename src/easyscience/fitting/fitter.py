# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from .available_minimizers import AvailableMinimizers
from .available_minimizers import from_string_to_enum
from .minimizers import FitResults
from .minimizers import MinimizerBase
from .minimizers.factory import factory

DEFAULT_MINIMIZER = AvailableMinimizers.LMFit_leastsq


class Fitter:
    """
    Fitter is a class which makes it possible to undertake fitting
    utilizing one of the supported minimizers.
    """

    def __init__(self, fit_object, fit_function: Callable):
        self._fit_object = fit_object
        self._fit_function = fit_function
        self._dependent_dims: int = None
        self._tolerance: float = None
        self._max_evaluations: int = None

        self._minimizer: MinimizerBase = None  # set in _update_minimizer
        self._enum_current_minimizer: AvailableMinimizers = None  # set in _update_minimizer
        self._update_minimizer(DEFAULT_MINIMIZER)

    def make_model(self, pars=None) -> Callable:
        return self._minimizer.make_model(pars)

    def evaluate(self, pars=None) -> np.ndarray:
        return self._minimizer.evaluate(pars)

    def convert_to_pars_obj(self, pars) -> object:
        return self._minimizer.convert_to_pars_obj(pars)

    # TODO: remove this method when we are ready to adjust the dependent products
    def initialize(self, fit_object: object, fit_function: Callable) -> None:
        """
        Set the model and callable in the calculator interface.

        Parameters
        ----------
        fit_object : object
            The EasyScience model object.
        fit_function : Callable
            The function to be optimized against.
        """
        self._fit_object = fit_object
        self._fit_function = fit_function
        self._update_minimizer(DEFAULT_MINIMIZER)

    # TODO: remove this method when we are ready to adjust the dependent products
    def create(self, minimizer_enum: Union[AvailableMinimizers, str] = DEFAULT_MINIMIZER) -> None:
        """
        Create the required minimizer.

        Parameters
        ----------
        minimizer_enum : Union[AvailableMinimizers, str], default=DEFAULT_MINIMIZER
            The enum of the minimization engine to create. By default,
            DEFAULT_MINIMIZER.
        """
        if isinstance(minimizer_enum, str):
            print(f'minimizer should be set with enum {minimizer_enum}')
            minimizer_enum = from_string_to_enum(minimizer_enum)
        self._update_minimizer(minimizer_enum)

    def switch_minimizer(self, minimizer_enum: Union[AvailableMinimizers, str]) -> None:
        """
        Switch minimizer and initialize.

        Parameters
        ----------
        minimizer_enum : Union[AvailableMinimizers, str]
            The enum of the minimizer to create and instantiate.
        """
        if isinstance(minimizer_enum, str):
            print(f'minimizer should be set with enum {minimizer_enum}')
            minimizer_enum = from_string_to_enum(minimizer_enum)

        self._update_minimizer(minimizer_enum)

    def _update_minimizer(self, minimizer_enum: AvailableMinimizers) -> None:
        self._minimizer = factory(
            minimizer_enum=minimizer_enum,
            fit_object=self._fit_object,
            fit_function=self.fit_function,
        )
        self._enum_current_minimizer = minimizer_enum

    @property
    def available_minimizers(self) -> List[str]:
        """
        Get a list of the names of available fitting minimizers.

        Returns
        -------
        List[str]
            List of available fitting minimizers.
        """
        return [minimize.name for minimize in AvailableMinimizers]

    @property
    def minimizer(self) -> MinimizerBase:
        """
        Get the current fitting minimizer object.

        Returns
        -------
        MinimizerBase
        """
        return self._minimizer

    @property
    def tolerance(self) -> float:
        """
        Get the tolerance for the minimizer.

        Returns
        -------
        float
            Tolerance for the minimizer.
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, tolerance: float) -> None:
        """
        Set the tolerance for the minimizer.

        Parameters
        ----------
        tolerance : float
            Tolerance for the minimizer.
        """
        self._tolerance = tolerance

    @property
    def max_evaluations(self) -> int:
        """
        Get the maximal number of evaluations for the minimizer.

        Returns
        -------
        int
            Maximal number of steps for the minimizer.
        """
        return self._max_evaluations

    @max_evaluations.setter
    def max_evaluations(self, max_evaluations: int) -> None:
        """
        Set the maximal number of evaluations for the minimizer.

        Parameters
        ----------
        max_evaluations : int
            Maximal number of steps for the minimizer.
        """
        self._max_evaluations = max_evaluations

    @property
    def fit_function(self) -> Callable:
        """
        Get the raw fit function that the optimizer will call.

        Returns
        -------
        Callable
            Raw fit function.
        """
        return self._fit_function

    @fit_function.setter
    def fit_function(self, fit_function: Callable) -> None:
        """
        Set the raw fit function to a new one.

        Parameters
        ----------
        fit_function : Callable
            New fit function.

        Returns
        -------
        None
            None.
        """
        self._fit_function = fit_function
        self._update_minimizer(self._enum_current_minimizer)

    @property
    def fit_object(self) -> object:
        """
        Get the EasyScience object used as a model.

        Returns
        -------
        object
            EasyScience model object.
        """
        return self._fit_object

    @fit_object.setter
    def fit_object(self, fit_object: object) -> None:
        """
        Set the EasyScience object which wil be used as a model.

        Parameters
        ----------
        fit_object : object
            New EasyScience object.
        """
        self._fit_object = fit_object
        self._update_minimizer(self._enum_current_minimizer)

    def _fit_function_wrapper(
        self, real_x: Optional[np.ndarray] = None, flatten: bool = True
    ) -> Callable:
        """
        Simple fit function which injects the real X (independent)
        values into the optimizer function.

        This will also flatten the results if needed.

        Parameters
        ----------
        real_x : Optional[np.ndarray], default=None
            Independent x parameters to be injected. By default, None.
        flatten : bool, default=True
            Should the result be a flat 1D array? By default, True.

        Returns
        -------
        Callable
            Wrapped optimizer function.
        """
        fun = self._fit_function

        @functools.wraps(fun)
        def wrapped_fit_function(x, **kwargs):
            if real_x is not None:
                x = real_x
            dependent = fun(x, **kwargs)
            if flatten:
                dependent = dependent.flatten()
            return dependent

        return wrapped_fit_function

    @property
    def fit(self) -> Callable:
        """
        Property which wraps the current ``fit`` function from the
        fitting interface.

        This property return a wrapped fit function which converts the
        input data into the correct shape for the optimizer, wraps the
        fit function to re-constitute the independent variables and once
        the fit is completed, reshape the inputs to those expected.
        """

        @functools.wraps(self._minimizer.fit)
        def inner_fit_callable(
            x: np.ndarray,
            y: np.ndarray,
            weights: Optional[np.ndarray] = None,
            vectorized: bool = False,
            progress_callback: Callable[[dict], bool | None] | None = None,
            **kwargs,
        ) -> FitResults:
            """
            This is a wrapped callable which performs the actual
            fitting. It is split into.

            3 sections, PRE/ FIT/ POST.
            - PRE = Reshaping the input data into the correct dimensions for the optimizer
            - FIT = Wrapping the fit function and performing the fit
            - POST = Reshaping the outputs so it is coherent with the inputs.
            """
            # Precompute - Reshape all independents into the correct dimensionality
            x_fit, x_new, y_new, weights, dims = self._precompute_reshaping(
                x, y, weights, vectorized
            )
            self._dependent_dims = dims

            # Fit
            fit_fun_org = self._fit_function
            fit_fun_wrap = self._fit_function_wrapper(
                x_new, flatten=True
            )  # This should be wrapped.
            self.fit_function = fit_fun_wrap
            f_res = self._minimizer.fit(
                x_fit,
                y_new,
                weights=weights,
                tolerance=self._tolerance,
                max_evaluations=self._max_evaluations,
                progress_callback=progress_callback,
                **kwargs,
            )

            # Postcompute
            fit_result = self._post_compute_reshaping(f_res, x, y)
            # Reset the function
            self.fit_function = fit_fun_org
            return fit_result

        return inner_fit_callable

    @staticmethod
    def _precompute_reshaping(
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray],
        vectorized: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], tuple[int, ...]]:
        """
        Check the dimensions of the inputs and reshape if necessary.

        Parameters
        ----------
        x : np.ndarray
            ND matrix of dependent points.
        y : np.ndarray
            N-1D matrix of independent points.
        weights : Optional[np.ndarray]
            Optional weights for the fit.
        vectorized : bool
            Whether ``x`` already stores vectorized coordinates.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], tuple[int, ...]]
            Reshaped x values, reshaped input data, flattened y values,
            flattened weights, and the original x shape.

        Raises
        ------
        ValueError
            If the shapes of ``x`` and ``y`` are incompatible.
        """
        # Make sure that they are np arrays
        x_new = np.array(x)
        y_new = np.array(y)
        # Get the shape
        x_shape = x_new.shape
        # Check if the x data is 1D
        if len(x_shape) > 1:
            # It is ND data
            # Check if the data is vectorized. i.e. should x be [NxMx...x Ndims]
            if vectorized:
                # Assert that the shapes are the same
                if np.all(x_shape[:-1] != y_new.shape):
                    raise ValueError('The shape of the x and y data must be the same')
                # If so do nothing but note that the data is vectorized
                # x_shape = (-1,) # Should this be done?
            else:
                # Assert that the shapes are the same
                if np.prod(x_new.shape[:-1]) != y_new.size:
                    raise ValueError('The number of elements in x and y data must be the same')
                # Reshape the data to be [len(NxMx..), Ndims] i.e. flatten to columns
                x_new = x_new.reshape(-1, x_shape[-1], order='F')
        else:
            # Assert that the shapes are the same
            if np.all(x_shape != y_new.shape):
                raise ValueError('The shape of the x and y data must be the same')
            # It is 1D data
            x_new = x.flatten()
        # The optimizer needs a 1D array, flatten the y data
        y_new = y_new.flatten()
        if weights is not None:
            weights = np.array(weights).flatten()
        # Make a 'dummy' x array for the fit function
        x_for_fit = np.array(range(y_new.size))
        return x_for_fit, x_new, y_new, weights, x_shape

    @staticmethod
    def _post_compute_reshaping(
        fit_result: FitResults, x: np.ndarray, y: np.ndarray
    ) -> FitResults:
        """
        Reshape the output of the fitter into the correct dimensions.

        Parameters
        ----------
        fit_result : FitResults
            Output from the fitter.
        x : np.ndarray
            Input x independent.
        y : np.ndarray
            Input y dependent.

        Returns
        -------
        FitResults
            Reshaped Fit Results.
        """
        fit_result.x = x
        fit_result.y_obs = y
        fit_result.y_calc = np.reshape(fit_result.y_calc, y.shape)
        fit_result.y_err = np.reshape(fit_result.y_err, y.shape)
        return fit_result

    def mcmc_sample(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        samples: int = 10000,
        burn: int = 2000,
        thin: int = 10,
        population: Optional[int] = None,
        seed: Optional[int] = None,
        resume_state: Optional[Any] = None,
        vectorized: bool = False,
        sampler_kwargs: Optional[dict] = None,
        progress_callback: Optional[Callable[[dict], Optional[bool]]] = None,
        abort_test: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Run Bayesian MCMC sampling using the BUMPS DREAM sampler.

        Works with both a plain ``Fitter`` (single dataset) and a
        ``MultiFitter`` (multiple datasets) via polymorphic dispatch:
        ``_precompute_reshaping`` and ``_fit_function_wrapper`` are resolved
        on the concrete subclass at call time, so multi-dataset flattening
        is handled automatically when called on a ``MultiFitter`` instance.

        Parameters
        ----------
        x : np.ndarray
            Independent variable array (or list of arrays for ``MultiFitter``).
        y : np.ndarray
            Dependent variable array (or list of arrays for ``MultiFitter``).
        weights : np.ndarray
            Weight array (or list of arrays for ``MultiFitter``).
        samples : int, default=10000
            Number of retained DREAM samples requested from BUMPS.
        burn : int, default=2000
            Burn-in steps to discard before collecting samples.
        thin : int, default=10
            Thinning interval — only every ``thin``-th sample is kept,
            which reduces autocorrelation between consecutive draws.
        population : Optional[int], default=None
            DREAM population **scale factor** (not an absolute chain count):
            BUMPS creates ``ceil(population * n_parameters)`` parallel chains.
        seed : Optional[int], default=None
            Best-effort random seed.  Calls ``numpy.random.seed(seed)``
            before DREAM starts, which affects the *global* NumPy RNG
            state.  Ignored when ``resume_state`` is provided (the saved
            chain has already advanced the RNG state).
        resume_state : Optional[Any], default=None
            A BUMPS ``MCMCDraw`` state object from a previous
            ``mcmc_sample()`` call (the ``'internal_bumps_object'`` value
            of the returned dict).  When provided, DREAM **continues** the
            saved chain instead of starting cold.  The population, parameter
            count, and parameter names must match the current model.

            **Ring-buffer contract:** DREAM stores draws in a fixed-size
            ring buffer sized to *samples*.  Resuming with ``samples=N``
            retains only the last N draws.  To extend an existing chain of
            M draws by N without losing any::

                fitter.mcmc_sample(
                    data, samples=M + N, burn=0, resume_state=previous_state
                )

            The ``burn`` parameter controls burn-in for the *new* draws
            only; passing ``burn=0`` (strongly recommended on resume)
            skips additional burn-in.  A non-zero ``burn`` on a
            previously-converged chain is usually a mistake.

            Resuming against *different* data is undefined behaviour (the
            chain's likelihood changes underneath it).
        vectorized : bool, default=False
            When ``True``, each x array may be multi-dimensional (e.g. an
            ``(N, M, 2)`` grid for a 2D model) and is left as-is.  When
            ``False`` (default), each x array is expected to be 1-D.
        sampler_kwargs : Optional[dict], default=None
            Additional keyword arguments forwarded to the BUMPS DREAM sampler.
        progress_callback : Optional[Callable[[dict], Optional[bool]]], default=None
            Optional callback invoked at each DREAM generation.  The payload
            dict includes ``iteration`` and ``sampling: True``.
        abort_test : Optional[Callable[[], bool]], default=None
            Optional callable that returns ``True`` to abort sampling early.

        Returns
        -------
        dict
            Dictionary with keys ``'draws'``, ``'param_names'``,
            ``'internal_bumps_object'``, and ``'logp'``.

        Raises
        ------
        ValueError
            If ``samples``, ``burn``, or ``thin`` are invalid.
        RuntimeError
            If the active minimizer is not a BUMPS instance.
        """
        if not isinstance(samples, int) or samples <= 0:
            raise ValueError('samples must be a positive integer.')
        if not isinstance(burn, int) or burn < 0:
            raise ValueError('burn must be a non-negative integer.')
        if not isinstance(thin, int) or thin < 1:
            raise ValueError('thin must be a positive integer.')

        x_fit, x_new, y_new, w_new, dims = self._precompute_reshaping(x, y, weights, vectorized)
        self._dependent_dims = dims

        original_fit_func = self._fit_function
        self.fit_function = self._fit_function_wrapper(x_new, flatten=True)

        try:
            minimizer = self.minimizer
            if not (hasattr(minimizer, 'package') and minimizer.package == 'bumps'):
                raise RuntimeError(
                    'Bayesian sampling requires a BUMPS minimizer. '
                    'Use ``fitter.switch_minimizer(AvailableMinimizers.Bumps)`` first.'
                )

            result = minimizer.mcmc_sample(
                x=x_fit,
                y=y_new,
                weights=w_new,
                samples=samples,
                burn=burn,
                thin=thin,
                population=population,
                seed=seed,
                resume_state=resume_state,
                sampler_kwargs=sampler_kwargs,
                progress_callback=progress_callback,
                abort_test=abort_test,
            )
        finally:
            self.fit_function = original_fit_func

        return result
