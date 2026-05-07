# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import Any
from typing import Callable
from typing import List

import numpy as np
from lmfit import Model as LMModel
from lmfit import Parameter as LMParameter
from lmfit import Parameters as LMParameters
from lmfit.model import ModelResult

# causes circular import when Parameter is imported
# from easyscience.base_classes import ObjBase
from easyscience.variable import Parameter

from ..available_minimizers import AvailableMinimizers
from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults


class LMFit(MinimizerBase):  # noqa: S101
    """
    This is a wrapper to the extended Levenberg-Marquardt Fit:
    https://lmfit.github.io/lmfit-py/ It allows for the lmfit fitting
    engine to use parameters declared in an
    ``EasyScience.base_classes.ObjBase``.
    """

    package = 'lmfit'

    def __init__(
        self,
        obj: object,  #: ObjBase,
        fit_function: Callable,
        minimizer_enum: AvailableMinimizers | None = None,
    ):  # todo after constraint changes, add type hint: obj: ObjBase  # noqa: E501
        """
        Initialize the minimizer.

        Parameters
        ----------
        obj : object
            Object containing the ``Parameter`` instances to fit.
        fit_function : Callable
            Callable returning model y values for the supplied x values.
        minimizer_enum : AvailableMinimizers | None, default=None
            Selected LMFit minimizer configuration. By default, None.
        """
        super().__init__(obj=obj, fit_function=fit_function, minimizer_enum=minimizer_enum)
        self._last_iteration: int | None = None

    @staticmethod
    def all_methods() -> List[str]:
        return [
            'least_squares',
            'leastsq',
            'differential_evolution',
            'basinhopping',
            'ampgo',
            'nelder',
            'lbfgsb',
            'powell',
            'cg',
            'newton',
            'cobyla',
            'bfgs',
        ]

    @staticmethod
    def supported_methods() -> List[str]:
        """Supported methods."""
        return [
            'least_squares',
            'leastsq',
            'differential_evolution',
            'powell',
            'cobyla',
        ]

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray = None,
        model: LMModel | None = None,
        parameters: LMParameters | None = None,
        method: str | None = None,
        tolerance: float | None = None,
        max_evaluations: int | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        minimizer_kwargs: dict | None = None,
        engine_kwargs: dict | None = None,
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit using the lmfit engine.

        Parameters
        ----------
        x : np.ndarray
            Points to be calculated at.
        y : np.ndarray
            Measured points.
        weights : np.ndarray, default=None
            Weights for supplied measured points. By default, None.
        model : LMModel | None, default=None
            Optional Model which is being fitted to. By default, None.
        parameters : LMParameters | None, default=None
            Optional parameters for the fit. By default, None.
        method : str | None, default=None
            Minimizer method. By default, None.
        tolerance : float | None, default=None
            Requested optimizer tolerance. By default, None.
        max_evaluations : int | None, default=None
            Maximum number of function evaluations. By default, None.
        progress_callback : Callable[[dict], bool | None] | None, default=None
            Optional callback receiving normalized progress payloads.
        minimizer_kwargs : dict | None, default=None
            Additional keyword arguments passed to LMFit's minimizer. By
            default, None.
        engine_kwargs : dict | None, default=None
            Additional engine keyword arguments. By default, None.
        **kwargs :
            Additional arguments for the fitting function.

        Returns
        -------
        FitResults
            Fit results should be 1/sigma, where sigma is the standard
            deviation of the measurement. For unweighted least squares,
            these should be 1.

        Raises
        ------
        FitError
            If the LMFit optimization fails.
        ValueError
            If the input shapes or weights are invalid.
        """
        x, y, weights = np.asarray(x), np.asarray(y), np.asarray(weights)

        if y.shape != x.shape:
            raise ValueError('x and y must have the same shape.')

        if weights.shape != x.shape:
            raise ValueError('Weights must have the same shape as x and y.')

        if not np.isfinite(weights).all():
            raise ValueError('Weights cannot be NaN or infinite.')

        if (weights <= 0).any():
            raise ValueError('Weights must be strictly positive and non-zero.')

        if engine_kwargs is None:
            engine_kwargs = {}

        method_kwargs = self._get_method_kwargs(method)
        fit_kws_dict = self._get_fit_kws(method, tolerance, minimizer_kwargs)

        # Why do we do this? Because a fitting template has to have global_object instantiated outside pre-runtime
        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False

        try:
            if model is None:
                model = self._make_model()

            self._last_iteration = None
            iter_cb = self._create_iter_callback(progress_callback)
            model_results = model.fit(
                y,
                x=x,
                weights=weights,
                max_nfev=max_evaluations,
                iter_cb=iter_cb,
                fit_kws=fit_kws_dict,
                **method_kwargs,
                **engine_kwargs,
                **kwargs,
            )
            self._set_parameter_fit_result(model_results, stack_status)
            results = self._gen_fit_results(model_results, iterations=self._last_iteration)
        except Exception as e:
            self._restore_parameter_values()
            raise FitError(e)
        finally:
            global_object.stack.enabled = stack_status
        return results

    def _create_iter_callback(
        self,
        progress_callback: Callable[[dict], bool | None] | None,
    ) -> Callable | None:
        """Create iter callback."""

        def iter_cb(params, iteration: int, residuals: np.ndarray, *args, **kwargs) -> bool:
            """Iter cb."""
            if iteration >= 0:
                self._last_iteration = int(iteration)
            if progress_callback is None:
                return False
            payload = self._build_progress_payload(params, iteration, residuals)
            progress_callback(payload)
            return False

        return iter_cb

    def _build_progress_payload(self, params, iteration: int, residuals: np.ndarray) -> dict:
        """Build progress payload."""
        residual_array = np.asarray(residuals)
        chi2 = float(np.square(residual_array).sum())
        varied_parameter_count = sum(
            1 for parameter in params.values() if getattr(parameter, 'vary', False)
        )
        degrees_of_freedom = residual_array.size - varied_parameter_count
        reduced_chi2 = chi2 / degrees_of_freedom if degrees_of_freedom > 0 else chi2

        parameter_values = {
            parameter_name[len(MINIMIZER_PARAMETER_PREFIX) :]: float(parameter.value)
            for parameter_name, parameter in params.items()
            if parameter_name.startswith(MINIMIZER_PARAMETER_PREFIX)
        }
        for parameter_name, parameter in self._cached_pars.items():
            lmfit_parameter_name = f'{MINIMIZER_PARAMETER_PREFIX}{parameter_name}'
            if lmfit_parameter_name not in params:
                parameter_values[parameter_name] = float(parameter.value)

        return {
            'iteration': int(iteration),
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
            'parameter_values': parameter_values,
            'refresh_plots': False,
            'finished': False,
        }

    def _get_fit_kws(
        self, method: str, tolerance: float, minimizer_kwargs: dict[str:str]
    ) -> dict[str:str]:
        """Get fit kws."""
        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        if tolerance is not None:
            if method in [None, 'least_squares', 'leastsq']:
                minimizer_kwargs['ftol'] = tolerance
            if method in ['differential_evolution', 'powell', 'cobyla']:
                minimizer_kwargs['tol'] = tolerance
        return minimizer_kwargs

    def convert_to_pars_obj(self, parameters: List[Parameter] | None = None) -> LMParameters:
        """
        Create an lmfit compatible container with the ``Parameters``
        converted from the base object.

        Parameters
        ----------
        parameters : List[Parameter] | None, default=None
            If only a single/selection of parameter is required. Specify
            as a list. By default, None.

        Returns
        -------
        LMParameters
            Lmfit Parameters compatible object.
        """
        if parameters is None:
            # Assume that we have a ObjBase for which we can obtain a list
            parameters = self._object.get_fit_parameters()
        lm_parameters = LMParameters().add_many([
            self.convert_to_par_object(parameter) for parameter in parameters
        ])
        return lm_parameters

    @staticmethod
    def convert_to_par_object(parameter: Parameter) -> LMParameter:
        """
        Convert an EasyScience Parameter object to a lmfit Parameter
        object.

        Parameters
        ----------
        parameter : Parameter
            EasyScience parameter to convert.

        Returns
        -------
        LMParameter
            Lmfit Parameter compatible object.
        """
        value = parameter.value

        return LMParameter(
            MINIMIZER_PARAMETER_PREFIX + parameter.unique_name,
            value=value,
            vary=not parameter.fixed,
            min=parameter.min,
            max=parameter.max,
            expr=None,
            brute_step=None,
        )

    def _make_model(self, pars: LMParameters | None = None) -> LMModel:
        """
        Generate a lmfit model from the supplied ``fit_function`` and
        parameters in the base object.

        Parameters
        ----------
        pars : LMParameters | None, default=None
            Optional lmfit parameter container.

        Returns
        -------
        LMModel
            Callable lmfit model.
        """
        # Generate the fitting function
        fit_func = self._generate_fit_function()

        self._fit_function = fit_func

        if pars is None:
            pars = self._cached_pars
        # Create the model
        model = LMModel(
            fit_func,
            independent_vars=['x'],
            param_names=[MINIMIZER_PARAMETER_PREFIX + str(key) for key in pars.keys()],
        )
        # Assign values from the `Parameter` to the model
        for name, item in pars.items():
            if isinstance(item, LMParameter):
                value = item.value
            else:
                value = item.value

            model.set_param_hint(
                MINIMIZER_PARAMETER_PREFIX + str(name),
                value=value,
                min=item.min,
                max=item.max,
            )

        # Cache the model for later reference
        self._cached_model = model
        return model

    def _set_parameter_fit_result(self, fit_result: ModelResult, stack_status: bool) -> None:
        """
        Update parameters to their final values and assign a std error
        to them.

        Parameters
        ----------
        fit_result : ModelResult
            Fit object which contains info on the fit.
        stack_status : bool
            Whether the undo stack was enabled.
        """
        from easyscience import global_object

        pars = self._cached_pars
        if stack_status:
            self._restore_parameter_values()
            global_object.stack.enabled = True
            global_object.stack.beginMacro('Fitting routine')
        for name in pars.keys():
            pars[name].value = fit_result.params[MINIMIZER_PARAMETER_PREFIX + str(name)].value
            if fit_result.errorbars:
                pars[name].error = fit_result.params[MINIMIZER_PARAMETER_PREFIX + str(name)].stderr
            else:
                pars[name].error = 0.0
        if stack_status:
            global_object.stack.endMacro()

    def _gen_fit_results(self, fit_results: ModelResult, **kwargs: Any) -> FitResults:
        """
        Convert fit results into the unified ``FitResults`` format.

        See
        https://github.com/lmfit/lmfit-py/blob/480072b9f7834b31ff2ca66277a5ad31246843a4/lmfit/model.py#L1272

        Parameters
        ----------
        fit_results : ModelResult
            Fit object which contains info on the fit.
        **kwargs : Any
            Additional result attributes to copy onto ``FitResults``.

        Returns
        -------
        FitResults
            Fit results container.
        """
        results = FitResults()
        for name, value in kwargs.items():
            if getattr(results, name, False):
                setattr(results, name, value)

        # We need to unify return codes......
        results.success = fit_results.success
        results.y_obs = fit_results.data
        # results.residual = fit_results.residual
        results.x = fit_results.userkws['x']
        results.p = fit_results.values
        results.p0 = fit_results.init_values
        # results.goodness_of_fit = fit_results.chisqr
        results.y_calc = fit_results.best_fit
        results.y_err = 1 / fit_results.weights
        results.n_evaluations = fit_results.nfev
        results.iterations = kwargs.get('iterations')
        results.message = fit_results.message
        if fit_results.success is False and fit_results.message:
            warnings.warn(str(fit_results.message), UserWarning)
        results.minimizer_engine = self.__class__
        results.fit_args = None

        results.engine_result = fit_results
        # results.check_sanity()
        return results
