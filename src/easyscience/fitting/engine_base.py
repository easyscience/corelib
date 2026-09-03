# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from inspect import Parameter as InspectParameter
from inspect import Signature
from inspect import _empty
from typing import Callable
from typing import Dict
from typing import Tuple

import numpy as np

from easyscience.variable import Parameter

PARAMETER_PREFIX = 'p'


class EngineBase(metaclass=ABCMeta):
    """
    Base for all evaluation engines: minimizers and samplers.

    An engine binds an EasyScience object and a fit function, and
    repeatedly evaluates the function while writing values back into the
    object's ``Parameter`` instances. ``EngineBase`` owns this shared
    system: the parameter cache, the ``Parameter``-writing wrapped
    fit function, and value restore on failure. It deliberately declares
    no abstract methods: the interfaces are defined on its subclasses
    (``MinimizerBase.fit``, ``DreamSampler.run``).
    """

    package: str = None

    def __init__(
        self,
        obj,
        fit_function: Callable,
    ):
        self._object = obj
        self._original_fit_function = fit_function
        self._cached_pars: Dict[str, Parameter] = {}
        self._cached_pars_vals: Dict[str, Tuple[float, float]] = {}
        self._fit_function = None

    def _restore_parameter_values(self) -> None:
        for key in self._cached_pars.keys():
            self._cached_pars[key].value = self._cached_pars_vals[key][0]
            self._cached_pars[key].error = self._cached_pars_vals[key][1]

    def evaluate(
        self, x: np.ndarray, minimizer_parameters: dict[str, float] | None = None, **kwargs
    ) -> np.ndarray:
        """
        Evaluate the fit function for values of x.

        Parameters used are either the latest or user supplied. If the
        parameters are user supplied, it must be in a dictionary of
        {'parameter_name': parameter_value,...}.

        Parameters
        ----------
        x : np.ndarray
            X values for which the fit function will be evaluated.
        minimizer_parameters : dict[str, float] | None, default=None
            Dictionary of parameters which will be used in the fit
            function. They must be in a dictionary of {'parameter_name':
            parameter_value,...}. By default, None.
        **kwargs :
            Additional arguments.

        Returns
        -------
        np.ndarray
            Y values calculated at points x for a set of parameters.

        Raises
        ------
        TypeError
            If ``minimizer_parameters`` is not a dictionary.
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

    def _prepare_parameters(self, parameters: dict[str, float]) -> dict[str, float]:
        """
        Prepare the parameters for the engine.

        Parameters
        ----------
        parameters : dict[str, float]
            Dict of parameters for the engine with names as keys.

        Returns
        -------
        dict[str, float]
            Completed parameter dictionary for the engine.
        """
        pars = self._cached_pars

        for name, item in pars.items():
            parameter_name = PARAMETER_PREFIX + str(name)
            if parameter_name not in parameters.keys():
                parameters[parameter_name] = item.value
        return parameters

    def _generate_fit_function(self) -> Callable:
        """
        Using the user supplied ``fit_function``, wrap it in such a way
        we can update ``Parameter`` on iterations.

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
        def _fit_function(x: np.ndarray, **kwargs) -> np.ndarray:
            """
            Wrapped fit function which now has an EasyScience compatible
            form.

            Parameters
            ----------
            x : np.ndarray
                Array of data points to be calculated.
            **kwargs :
                Key word arguments.

            Returns
            -------
            np.ndarray
                Points calculated at ``x``.
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
        """
        Wrap the function signature.

        This is done as lmfit wants the function to be in the form: f =
        (x, a=1, b=2)... Where we need to be generic. Note that this
        won't hold for much outside of this scope.
        """
        wrapped_parameters = []
        wrapped_parameters.append(
            InspectParameter('x', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty)
        )

        for name, parameter in parameters.items():
            default_value = parameter.value

            wrapped_parameters.append(
                InspectParameter(
                    PARAMETER_PREFIX + str(name),
                    InspectParameter.POSITIONAL_OR_KEYWORD,
                    annotation=_empty,
                    default=default_value,
                )
            )
        return Signature(wrapped_parameters)
