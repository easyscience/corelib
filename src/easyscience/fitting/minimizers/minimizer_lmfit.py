#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience


from typing import Callable
from typing import List
from typing import Optional

import numpy as np
from lmfit import Model as LMModel
from lmfit import Parameter as LMParameter
from lmfit import Parameters as LMParameters
from lmfit.model import ModelResult

# causes circular import when Parameter is imported
# from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.variable import Parameter

from ..available_minimizers import AvailableMinimizers
from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults


class LMFit(MinimizerBase):  # noqa: S101
    """
    This is a wrapper to the extended Levenberg-Marquardt Fit: https://lmfit.github.io/lmfit-py/
    It allows for the lmfit fitting engine to use parameters declared in an `EasyScience.Objects.Base.BaseObj`.
    """

    package = 'lmfit'

    def __init__(
        self,
        obj,  #: BaseObj,
        fit_function: Callable,
        minimizer_enum: Optional[AvailableMinimizers] = None,
    ):  # todo after constraint changes, add type hint: obj: BaseObj  # noqa: E501
        """
        Initialize the minimizer with the `BaseObj` and the `fit_function` to be used.

        :param obj: Base object which contains the parameters to be fitted
        :type obj: BaseObj
        :param fit_function: Function which will be fitted to the data
        :type fit_function: Callable
        :param method: Method to be used by the minimizer
        :type method: str
        """
        super().__init__(obj=obj, fit_function=fit_function, minimizer_enum=minimizer_enum)

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
        weights: Optional[np.ndarray] = None,
        model: Optional[LMModel] = None,
        parameters: Optional[LMParameters] = None,
        method: Optional[str] = None,
        tolerance: Optional[float] = None,
        max_evaluations: Optional[int] = None,
        minimizer_kwargs: Optional[dict] = None,
        engine_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit using the lmfit engine.

        :param method:
        :type method:
        :param x: points to be calculated at
        :type x: np.ndarray
        :param y: measured points
        :type y: np.ndarray
        :param weights: Weights for supplied measured points
        :type weights: np.ndarray
        :param model: Optional Model which is being fitted to
        :type model: LMModel
        :param parameters: Optional parameters for the fit
        :type parameters: LMParameters
        :param minimizer_kwargs: Arguments to be passed directly to the minimizer
        :type minimizer_kwargs: dict
        :param kwargs: Additional arguments for the fitting function.
        :return: Fit results
        :rtype: ModelResult
        """
        if weights is None:
            weights = 1 / np.sqrt(np.abs(y))

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

            model_results = model.fit(
                y,
                x=x,
                weights=weights,
                max_nfev=max_evaluations,
                fit_kws=fit_kws_dict,
                **method_kwargs,
                **engine_kwargs,
                **kwargs,
            )
            self._set_parameter_fit_result(model_results, stack_status)
            results = self._gen_fit_results(model_results)
        except Exception as e:
            for key in self._cached_pars.keys():
                self._cached_pars[key].value = self._cached_pars_vals[key][0]
            raise FitError(e)
        return results

    def _get_fit_kws(self, method: str, tolerance: float, minimizer_kwargs: dict[str:str]) -> dict[str:str]:
        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        if tolerance is not None:
            if method in [None, 'least_squares', 'leastsq']:
                minimizer_kwargs['ftol'] = tolerance
            if method in ['differential_evolution', 'powell', 'cobyla']:
                minimizer_kwargs['tol'] = tolerance
        return minimizer_kwargs

    def convert_to_pars_obj(self, parameters: Optional[List[Parameter]] = None) -> LMParameters:
        """
        Create an lmfit compatible container with the `Parameters` converted from the base object.

        :param parameters: If only a single/selection of parameter is required. Specify as a list
        :return: lmfit Parameters compatible object
        """
        if parameters is None:
            # Assume that we have a BaseObj for which we can obtain a list
            parameters = self._object.get_fit_parameters()
        lm_parameters = LMParameters().add_many([self.convert_to_par_object(parameter) for parameter in parameters])
        return lm_parameters

    @staticmethod
    def convert_to_par_object(parameter: Parameter) -> LMParameter:
        """
        Convert an `EasyScience.Objects.Base.Parameter` object to a lmfit Parameter object.

        :return: lmfit Parameter compatible object.
        :rtype: LMParameter
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

    def _make_model(self, pars: Optional[LMParameters] = None) -> LMModel:
        """
        Generate a lmfit model from the supplied `fit_function` and parameters in the base object.

        :return: Callable lmfit model
        :rtype: LMModel
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

            model.set_param_hint(MINIMIZER_PARAMETER_PREFIX + str(name), value=value, min=item.min, max=item.max)

        # Cache the model for later reference
        self._cached_model = model
        return model

    def _set_parameter_fit_result(self, fit_result: ModelResult, stack_status: bool):
        """
        Update parameters to their final values and assign a std error to them.

        :param fit_result: Fit object which contains info on the fit
        :return: None
        :rtype: noneType
        """
        from easyscience import global_object

        pars = self._cached_pars
        if stack_status:
            for name in pars.keys():
                pars[name].value = self._cached_pars_vals[name][0]
                pars[name].error = self._cached_pars_vals[name][1]
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

    def _gen_fit_results(self, fit_results: ModelResult, **kwargs) -> FitResults:
        """
        Convert fit results into the unified `FitResults` format.
        See https://github.com/lmfit/lmfit-py/blob/480072b9f7834b31ff2ca66277a5ad31246843a4/lmfit/model.py#L1272

        :param fit_result: Fit object which contains info on the fit
        :return: fit results container
        :rtype: FitResults
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
        results.minimizer_engine = self.__class__
        results.fit_args = None

        results.engine_result = fit_results
        # results.check_sanity()
        return results
