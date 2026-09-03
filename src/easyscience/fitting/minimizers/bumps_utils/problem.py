# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""BUMPS problem construction shared by the ``Bumps`` minimizer and
``DreamSampler``.

These are free functions rather than ``Bumps`` methods so that any
:class:`~easyscience.fitting.engine_base.EngineBase`: a minimizer or a
sampler, can build a BUMPS ``Curve``/``FitProblem`` without inheriting
from the minimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from bumps.names import Curve
from bumps.names import FitProblem
from bumps.parameter import Parameter as BumpsParameter

from easyscience.variable import Parameter

from ...engine_base import PARAMETER_PREFIX
from .eval_counter import EvalCounter

if TYPE_CHECKING:
    from ...engine_base import EngineBase


def to_bumps_parameter(par: Parameter) -> BumpsParameter:
    """Convert an EasyScience ``Parameter`` to a prefixed ``BumpsParameter``.

    Parameters
    ----------
    par : Parameter
        EasyScience parameter to convert.

    Returns
    -------
    BumpsParameter
        Bumps Parameter compatible object, named
        ``PARAMETER_PREFIX + par.unique_name``.
    """
    return BumpsParameter(
        name=PARAMETER_PREFIX + par.unique_name,
        value=par.value,
        bounds=[par.min, par.max],
        fixed=par.fixed,
    )


def build_curve_problem(
    engine: 'EngineBase',
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    parameters: list[Parameter] | None = None,
) -> tuple[FitProblem, EvalCounter, Curve]:
    """Build a BUMPS ``FitProblem`` around an engine's wrapped fit function.

    Wraps ``engine._generate_fit_function()`` in an :class:`EvalCounter`,
    converts the engine's cached parameters (or the explicitly supplied
    ``parameters``) via :func:`to_bumps_parameter`, and assembles
    ``Curve(fit_func, x, y, dy=1/weights, **bumps_pars)`` into a
    ``FitProblem``.

    Parameters
    ----------
    engine : EngineBase
        The engine (minimizer or sampler) supplying the fit function and
        parameter cache.
    x : np.ndarray
        Independent variable array.
    y : np.ndarray
        Dependent variable array.
    weights : np.ndarray
        Weight array; converted to ``dy = 1 / weights``.
    parameters : list[Parameter] | None, default=None
        Optional explicit EasyScience parameters to bind into the model
        instead of the engine's cached parameters.

    Returns
    -------
    tuple[FitProblem, EvalCounter, Curve]
        The assembled problem, the evaluation counter wrapping the fit
        function, and the ``Curve`` model itself.
        The ``Curve`` is returned because ``FitProblem.fitness`` is
        deprecated in current BUMPS.
    """
    fit_func = EvalCounter(engine._generate_fit_function())

    bumps_pars = {}
    if not parameters:
        for name, par in engine._cached_pars.items():
            bumps_pars[PARAMETER_PREFIX + str(name)] = to_bumps_parameter(par)
    else:
        for par in parameters:
            bumps_pars[PARAMETER_PREFIX + par.unique_name] = to_bumps_parameter(par)

    curve = Curve(fit_func, x, y, dy=1 / weights, **bumps_pars)
    return FitProblem(curve), fit_func, curve


def parameter_names(problem: FitProblem) -> list[str]:
    """Return the problem's parameter names with the prefix stripped.

    Parameters
    ----------
    problem : FitProblem
        A BUMPS problem built by :func:`build_curve_problem`.

    Returns
    -------
    list[str]
        Parameter names in problem order, without ``PARAMETER_PREFIX``.
    """
    return [(p.name or '')[len(PARAMETER_PREFIX) :] for p in problem._parameters]


def parameter_snapshot(problem: FitProblem, point: np.ndarray | None) -> dict:
    """Snapshot the problem's parameter values as ``{name: value}``.

    Parameters
    ----------
    problem : FitProblem
        A BUMPS problem built by :func:`build_curve_problem`.
    point : np.ndarray | None
        Parameter values to report; when ``None`` the problem's current
        values (``problem.getp()``) are used.

    Returns
    -------
    dict
        Mapping of prefix-stripped parameter names to ``float`` values.
    """
    labels = problem.labels()
    values = problem.getp() if point is None else point
    snapshot = {}
    for label, value in zip(labels, values):
        snapshot[label[len(PARAMETER_PREFIX) :]] = float(value)
    return snapshot
