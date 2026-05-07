# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from inspect import Parameter as InspectParameter
from inspect import Signature
from inspect import _empty
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np

# causes circular import when Parameter is imported
# from easyscience.base_classes import ObjBase
from easyscience.variable import Parameter

from ..available_minimizers import AvailableMinimizers
from .utils import FitError
from .utils import FitResults

MINIMIZER_PARAMETER_PREFIX = 'p'


class MinimizerBase(metaclass=ABCMeta):
    """This template class is the basis for all minimizer engines in
    `EasyScience`.
    """

    package: str = None

    def __init__(
        self,
        obj,  #: ObjBase,
        fit_function: Callable,
        minimizer_enum: AvailableMinimizers,
    ):  # todo after constraint changes, add type hint: obj: ObjBase  # noqa: E501
        if minimizer_enum.method not in self.supported_methods():
            raise FitError(f'Method {minimizer_enum.method} not available in {self.__class__}')
        self._object = obj
        self._original_fit_function = fit_function
        self._minimizer_enum = minimizer_enum
        self._method = minimizer_enum.method
        self._cached_pars: Dict[str, Parameter] = {}
        self._cached_pars_vals: Dict[str, Tuple[float]] = {}
        self._cached_model = None
        self._fit_function = None

    @property
    def enum(self) -> AvailableMinimizers:
        """Enum function."""
        return self._minimizer_enum

    @property
    def name(self) -> str:
        """Name function."""
        return self._minimizer_enum.name

    def _restore_parameter_values(self) -> None:
        """Restore parameter values."""
        for key in self._cached_pars.keys():
            self._cached_pars[key].value = self._cached_pars_vals[key][0]
            self._cached_pars[key].error = self._cached_pars_vals[key][1]

    @abstractmethod
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        model: Callable | None = None,
        parameters: List[Parameter] | None = None,
        method: str | None = None,
        tolerance: float | None = None,
        max_evaluations: int | None = None,
        progress_callback: Callable[[dict], bool | None] | None = None,
        **kwargs,
    ) -> FitResults:
        """Perform a fit using the  engine.

        Parameters
        ----------
        progress_callback : Callable[[dict], bool | None] | None, optional
            By default, None.
        max_evaluations : int | None, optional
            By default, None.
        tolerance : float | None, optional
            By default, None.
        x : np.ndarray
            Points to be calculated at.
        y : np.ndarray
            Measured points.
        weights : np.ndarray
            Weights for supplied measured points.
        model : Callable | None, optional
            Optional Model which is being fitted to. By default, None.
        parameters : List[Parameter] | None, optional
            Optional parameters for the fit. By default, None.
        method : str | None, optional
            Method for the minimizer to use. By default, None.
        **kwargs :
            Additional arguments for the fitting function.

        Returns
        -------
        FitResults
            Fit results.
        """

    def evaluate(
        self, x: np.ndarray, minimizer_parameters: dict[str, float] | None = None, **kwargs
    ) -> np.ndarray:
        """Evaluate the fit function for values of x.

        Parameters used
are either the latest or user supplied. If the parameters are
        user supplied, it must be in a dictionary of {'parameter_name':
        parameter_value,...}.

        Parameters
        ----------
        x : np.ndarray
            X values for which the fit function will be evaluated.
        minimizer_parameters : dict[str, float] | None, optional
            Dictionary of parameters which will be used in the fit function. They must be in a dictionary
            of {'parameter_name': parameter_value,...}. By default, None.
        **kwargs :
            Additional arguments.

        Returns
        -------
        np.ndarray
            Y values calculated at points x for a set of parameters.
        """
        if minimizer_parameters is None:
            minimizer_parameters = {}
        if not isinstance(minimizer_parameters, dict):
            raise TypeError('minimizer_parameters must be a dictionary')

        if self._fit_function is None:
            # This will also generate self._cached_pars
            self._fit_function = self._generate_fit_function()

        minimizer_parameters = self._prepare_parameters(minimizer_parameters)

        return self._fit_function(x, **minimizer_parameters, **kwargs)

    def _get_method_kwargs(self, passed_method: str | None = None) -> dict[str, str]:
        """Get method kwargs."""
        if passed_method is not None:
            if passed_method not in self.supported_methods():
                raise FitError(f'Method {passed_method} not available in {self.__class__}')
            return {'method': passed_method}

        if self._method is not None:
            return {'method': self._method}

        return {}

    @abstractmethod
    def convert_to_pars_obj(self, par_list: List[Parameter] | None = None):
        """Create an engine compatible container with the `Parameters`
        converted from the base object.

        Parameters
        ----------
        par_list : List[Parameter] | None, optional
            If only a single/selection of parameter is
            required. Specify as a list. By default, None.

        Returns
        -------

            Engine Parameters compatible object.
        """

    @staticmethod
    @abstractmethod
    def supported_methods() -> List[str]:
        """Return a list of supported methods for the minimizer.

        Returns
        -------
        List[str]
            List of supported methods.
        """

    @staticmethod
    @abstractmethod
    def all_methods() -> List[str]:
        """Return a list of all available methods for the minimizer.

        Returns
        -------
        List[str]
            List of all available methods.
        """

    @staticmethod
    @abstractmethod
    def convert_to_par_object(obj):  # todo after constraint changes, add type hint: obj: ObjBase
        """Convert an `EasyScience.variable.Parameter` object to an
        engine Parameter object.
        """

    def _prepare_parameters(self, parameters: dict[str, float]) -> dict[str, float]:
        """Prepare the parameters for the minimizer.

        Parameters
        ----------
        parameters : dict[str, float]
            Dict of parameters for the minimizer with
            names as keys.
        """
        pars = self._cached_pars

        for name, item in pars.items():
            parameter_name = MINIMIZER_PARAMETER_PREFIX + str(name)
            if parameter_name not in parameters.keys():
                parameters[parameter_name] = item.value
        return parameters

    def _generate_fit_function(self) -> Callable:
        """Using the user supplied `fit_function`, wrap it in such a way
        we can update `Parameter` on iterations.

        Returns
        -------
        Callable
            A fit function which is compatible with bumps models.
        """
        # Original fit function
        func = self._original_fit_function
        # Get a list of `Parameters`
        self._cached_pars = {}
        self._cached_pars_vals = {}
        for parameter in self._object.get_fit_parameters():
            key = parameter.unique_name
            self._cached_pars[key] = parameter
            self._cached_pars_vals[key] = (parameter.value, parameter.error)

        # Make a new fit function
        def _fit_function(x: np.ndarray, **kwargs):
            """Wrapped fit function which now has an EasyScience
            compatible form.

            Parameters
            ----------
            x : np.ndarray
                Array of data points to be calculated.
            **kwargs :
                Key word arguments.

            Returns
            -------
            np.ndarray
                Points calculated at `x`.
            """
            # Update the `Parameter` values and the callback if needed
            # TODO THIS IS NOT THREAD SAFE :-(

            for name, value in kwargs.items():
                par_name = name[1:]
                if par_name in self._cached_pars.keys():
                    # This will take into account constraints
                    if self._cached_pars[par_name].value != value:
                        self._cached_pars[par_name].value = value

                    # Since we are calling the parameter fset will be called.
            # TODO Pre processing here
            return_data = func(x)
            # TODO Loading or manipulating data here
            return return_data

        _fit_function.__signature__ = self._create_signature(self._cached_pars)
        return _fit_function

    @staticmethod
    def _create_signature(parameters: Dict[int, Parameter]) -> Signature:
        """Wrap the function signature.

        This is done as lmfit wants the function to be in the form:
        f = (x, a=1, b=2)...
        Where we need to be generic. Note that this won't hold for much outside of this scope.
        """
        wrapped_parameters = []
        wrapped_parameters.append(
            InspectParameter('x', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty)
        )

        for name, parameter in parameters.items():
            default_value = parameter.value

            wrapped_parameters.append(
                InspectParameter(
                    MINIMIZER_PARAMETER_PREFIX + str(name),
                    InspectParameter.POSITIONAL_OR_KEYWORD,
                    annotation=_empty,
                    default=default_value,
                )
            )
        return Signature(wrapped_parameters)

    @staticmethod
    def _error_from_jacobian(
        jacobian: np.ndarray, residuals: np.ndarray, confidence: float = 0.95
    ) -> np.ndarray:
        """Error from jacobian."""
        from scipy import stats

        JtJi = np.linalg.inv(np.dot(jacobian.T, jacobian))
        # 1.96 is a 95% confidence value
        error_matrix = np.dot(
            JtJi,
            np.dot(jacobian.T, np.dot(np.diag(residuals**2), np.dot(jacobian, JtJi))),
        )

        z = 1 - ((1 - confidence) / 2)
        z = stats.norm.pdf(z)
        error_matrix = z * np.sqrt(error_matrix)
        return error_matrix
