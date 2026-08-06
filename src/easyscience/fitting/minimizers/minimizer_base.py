# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import List

import numpy as np

# causes circular import when Parameter is imported
# from easyscience.base_classes import ObjBase
from easyscience.variable import Parameter

from ..available_minimizers import AvailableMinimizers
from ..engine_base import PARAMETER_PREFIX
from ..engine_base import EngineBase
from .utils import FitError
from .utils import FitResults

# Back-compat alias: the canonical constant now lives in
# ``easyscience.fitting.engine_base`` and is shared by minimizers and
# samplers alike.
MINIMIZER_PARAMETER_PREFIX = PARAMETER_PREFIX


class MinimizerBase(EngineBase):
    """
    This template class is the basis for all minimizer engines in
    ``EasyScience``.
    """

    def __init__(
        self,
        obj,  #: ObjBase,
        fit_function: Callable,
        minimizer_enum: AvailableMinimizers,
    ):  # todo after constraint changes, add type hint: obj: ObjBase  # noqa: E501
        if minimizer_enum.method not in self.supported_methods():
            raise FitError(f'Method {minimizer_enum.method} not available in {self.__class__}')
        super().__init__(obj=obj, fit_function=fit_function)
        self._minimizer_enum = minimizer_enum
        self._method = minimizer_enum.method
        self._cached_model = None

    @property
    def enum(self) -> AvailableMinimizers:
        return self._minimizer_enum

    @property
    def name(self) -> str:
        return self._minimizer_enum.name

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
        """
        Perform a fit using the  engine.

        Parameters
        ----------
        x : np.ndarray
            Points to be calculated at.
        y : np.ndarray
            Measured points.
        weights : np.ndarray
            Weights for supplied measured points.
        model : Callable | None, default=None
            Optional Model which is being fitted to. By default, None.
        parameters : List[Parameter] | None, default=None
            Optional parameters for the fit. By default, None.
        method : str | None, default=None
            Method for the minimizer to use. By default, None.
        tolerance : float | None, default=None
            Requested convergence tolerance. By default, None.
        max_evaluations : int | None, default=None
            Maximum number of objective evaluations. By default, None.
        progress_callback : Callable[[dict], bool | None] | None, default=None
            Optional progress callback. By default, None.
        **kwargs :
            Additional arguments for the fitting function.

        Returns
        -------
        FitResults
            Fit results.
        """

    def _get_method_kwargs(self, passed_method: str | None = None) -> dict[str, str]:
        if passed_method is not None:
            if passed_method not in self.supported_methods():
                raise FitError(f'Method {passed_method} not available in {self.__class__}')
            return {'method': passed_method}

        if self._method is not None:
            return {'method': self._method}

        return {}

    @abstractmethod
    def convert_to_pars_obj(self, par_list: List[Parameter] | None = None) -> Any:
        """
        Create an engine compatible container with the ``Parameters``
        converted from the base object.

        Parameters
        ----------
        par_list : List[Parameter] | None, default=None
            If only a single/selection of parameter is required. Specify
            as a list. By default, None.

        Returns
        -------
        Any
            Engine Parameters compatible object.
        """

    @staticmethod
    @abstractmethod
    def supported_methods() -> List[str]:
        """
        Return a list of supported methods for the minimizer.

        Returns
        -------
        List[str]
            List of supported methods.
        """

    @staticmethod
    @abstractmethod
    def all_methods() -> List[str]:
        """
        Return a list of all available methods for the minimizer.

        Returns
        -------
        List[str]
            List of all available methods.
        """

    @staticmethod
    @abstractmethod
    def convert_to_par_object(obj):  # todo after constraint changes, add type hint: obj: ObjBase
        """
        Convert an ``EasyScience.variable.Parameter`` object to an
        engine Parameter object.
        """

    @staticmethod
    def _error_from_jacobian(
        jacobian: np.ndarray, residuals: np.ndarray, confidence: float = 0.95
    ) -> np.ndarray:
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
