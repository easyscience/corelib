#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

import copy
from typing import Callable
from typing import List
from typing import Optional

import numpy as np
from bumps.fitters import FIT_AVAILABLE_IDS
from bumps.fitters import fit as bumps_fit
from bumps.names import Curve
from bumps.names import FitProblem
from bumps.parameter import Parameter as BumpsParameter

# causes circular import when Parameter is imported
# from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.variable import Parameter

from ..available_minimizers import AvailableMinimizers
from .minimizer_base import MINIMIZER_PARAMETER_PREFIX
from .minimizer_base import MinimizerBase
from .utils import FitError
from .utils import FitResults

FIT_AVAILABLE_IDS_FILTERED = copy.copy(FIT_AVAILABLE_IDS)
# Considered experimental
FIT_AVAILABLE_IDS_FILTERED.remove('pt')


class Bumps(MinimizerBase):
    """
    This is a wrapper to Bumps: https://bumps.readthedocs.io/
    It allows for the Bumps fitting engine to use parameters declared in an `EasyScience.Objects.Base.BaseObj`.
    """

    package = 'bumps'

    def __init__(
        self,
        obj,  #: BaseObj,
        fit_function: Callable,
        minimizer_enum: Optional[AvailableMinimizers] = None,
    ):  # todo after constraint changes, add type hint: obj: BaseObj  # noqa: E501
        """
        Initialize the fitting engine with a `BaseObj` and an arbitrary fitting function.

        :param obj: Object containing elements of the `Parameter` class
        :type obj: BaseObj
        :param fit_function: function that when called returns y values. 'x' must be the first
                            and only positional argument. Additional values can be supplied by
                            keyword/value pairs
        :type fit_function: Callable
        """
        super().__init__(obj=obj, fit_function=fit_function, minimizer_enum=minimizer_enum)
        self._p_0 = {}

    @staticmethod
    def all_methods() -> List[str]:
        return FIT_AVAILABLE_IDS_FILTERED

    @staticmethod
    def supported_methods() -> List[str]:
        # only a small subset
        methods = ['scipy.leastsq', 'amoeba', 'newton', 'lm']
        return methods

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        model: Optional[Callable] = None,
        parameters: Optional[Parameter] = None,
        method: Optional[str] = None,
        tolerance: Optional[float] = None,
        max_evaluations: Optional[int] = None,
        minimizer_kwargs: Optional[dict] = None,
        engine_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit using the lmfit engine.

        :param x: points to be calculated at
        :type x: np.ndarray
        :param y: measured points
        :type y: np.ndarray
        :param weights: Weights for supplied measured points * Not really optional*
        :type weights: np.ndarray
        :param model: Optional Model which is being fitted to
        :type model: lmModel
        :param parameters: Optional parameters for the fit
        :type parameters: List[BumpsParameter]
        :param kwargs: Additional arguments for the fitting function.
        :param method: Method for minimization
        :type method: str
        :return: Fit results
        :rtype: ModelResult
        """
        method_dict = self._get_method_kwargs(method)

        if weights is None:
            weights = np.sqrt(np.abs(y))

        if engine_kwargs is None:
            engine_kwargs = {}

        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        minimizer_kwargs.update(engine_kwargs)

        if tolerance is not None:
            minimizer_kwargs['ftol'] = tolerance  # tolerance for change in function value
            minimizer_kwargs['xtol'] = tolerance  # tolerance for change in parameter value, could be an independent value
        if max_evaluations is not None:
            minimizer_kwargs['steps'] = max_evaluations

        if model is None:
            model_function = self._make_model(parameters=parameters)
            model = model_function(x, y, weights)
        self._cached_model = model

        self._p_0 = {f'p{key}': self._cached_pars[key].value for key in self._cached_pars.keys()}

        problem = FitProblem(model)
        # Why do we do this? Because a fitting template has to have global_object instantiated outside pre-runtime
        from easyscience import global_object

        stack_status = global_object.stack.enabled
        global_object.stack.enabled = False

        try:
            model_results = bumps_fit(problem, **method_dict, **minimizer_kwargs, **kwargs)
            self._set_parameter_fit_result(model_results, stack_status)
            results = self._gen_fit_results(model_results)
        except Exception as e:
            for key in self._cached_pars.keys():
                self._cached_pars[key].value = self._cached_pars_vals[key][0]
            raise FitError(e)
        return results

    def convert_to_pars_obj(self, par_list: Optional[List] = None) -> List[BumpsParameter]:
        """
        Create a container with the `Parameters` converted from the base object.

        :param par_list: If only a single/selection of parameter is required. Specify as a list
        :type par_list: List[str]
        :return: bumps Parameters list
        :rtype: List[BumpsParameter]
        """
        if par_list is None:
            # Assume that we have a BaseObj for which we can obtain a list
            par_list = self._object.get_fit_parameters()
        pars_obj = [self.__class__.convert_to_par_object(obj) for obj in par_list]
        return pars_obj

    # For some reason I have to double staticmethod :-/
    @staticmethod
    def convert_to_par_object(obj) -> BumpsParameter:
        """
        Convert an `EasyScience.Objects.Base.Parameter` object to a bumps Parameter object

        :return: bumps Parameter compatible object.
        :rtype: BumpsParameter
        """

        value = obj.value

        return BumpsParameter(
            name=MINIMIZER_PARAMETER_PREFIX + obj.unique_name,
            value=value,
            bounds=[obj.min, obj.max],
            fixed=obj.fixed,
        )

    def _make_model(self, parameters: Optional[List[BumpsParameter]] = None) -> Callable:
        """
        Generate a bumps model from the supplied `fit_function` and parameters in the base object.
        Note that this makes a callable as it needs to be initialized with *x*, *y*, *weights*

        :return: Callable to make a bumps Curve model
        :rtype: Callable
        """
        fit_func = self._generate_fit_function()

        def _outer(obj):
            def _make_func(x, y, weights):
                bumps_pars = {}
                if not parameters:
                    for name, par in obj._cached_pars.items():
                        bumps_pars[MINIMIZER_PARAMETER_PREFIX + str(name)] = obj.convert_to_par_object(par)
                else:
                    for par in parameters:
                        bumps_pars[MINIMIZER_PARAMETER_PREFIX + par.unique_name] = obj.convert_to_par_object(par)
                return Curve(fit_func, x, y, dy=weights, **bumps_pars)

            return _make_func

        return _outer(self)

    def _set_parameter_fit_result(self, fit_result, stack_status: bool):
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

        for index, name in enumerate(self._cached_model._pnames):
            dict_name = name[len(MINIMIZER_PARAMETER_PREFIX) :]
            pars[dict_name].value = fit_result.x[index]
            pars[dict_name].error = fit_result.dx[index]
        if stack_status:
            global_object.stack.endMacro()

    def _gen_fit_results(self, fit_results, **kwargs) -> FitResults:
        """
        Convert fit results into the unified `FitResults` format

        :param fit_result: Fit object which contains info on the fit
        :return: fit results container
        :rtype: FitResults
        """

        results = FitResults()
        for name, value in kwargs.items():
            if getattr(results, name, False):
                setattr(results, name, value)
        results.success = fit_results.success
        pars = self._cached_pars
        item = {}
        for index, name in enumerate(self._cached_model._pnames):
            dict_name = name[len(MINIMIZER_PARAMETER_PREFIX) :]

            item[name] = pars[dict_name].value

        results.p0 = self._p_0
        results.p = item
        results.x = self._cached_model.x
        results.y_obs = self._cached_model.y
        results.y_calc = self.evaluate(results.x, minimizer_parameters=results.p)
        results.y_err = self._cached_model.dy
        # results.residual = results.y_obs - results.y_calc
        # results.goodness_of_fit = np.sum(results.residual**2)
        results.minimizer_engine = self.__class__
        results.fit_args = None
        results.engine_result = fit_results
        # results.check_sanity()
        return results
