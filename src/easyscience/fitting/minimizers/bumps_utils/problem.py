# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""BUMPS problem construction shared by the ``Bumps`` minimizer and
``DreamSampler``.

These are free functions rather than ``Bumps`` methods so that any
:class:`~easyscience.fitting.engine_base.EngineBase` — a minimizer or a
sampler — can build a BUMPS ``Curve``/``FitProblem`` without inheriting
from the minimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Sequence

import numpy as np
from bumps.names import Curve
from bumps.names import FitProblem
from bumps.parameter import Parameter as BumpsParameter

from easyscience.variable import Parameter

from ...engine_base import PARAMETER_PREFIX
from .eval_counter import EvalCounter

if TYPE_CHECKING:
    from ...engine_base import EngineBase

#: Signature of the optional inequality-constraints hook: receives the
#: ``{prefixed name: BumpsParameter}`` mapping of a freshly built problem
#: and returns the constraint objects to attach to the ``FitProblem``.
ConstraintsFactory = Callable[[Dict[str, BumpsParameter]], Sequence[Any]]


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
    constraints_factory: ConstraintsFactory | None = None,
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
    constraints_factory : ConstraintsFactory | None, default=None
        Optional callable producing the BUMPS inequality constraints for
        this problem. It receives the freshly built mapping
        ``{PARAMETER_PREFIX + unique_name: BumpsParameter}`` of the *free*
        parameters (fixed and dependent EasyScience parameters are not part
        of the problem: treat them as constants, resp. expand them into
        their free leaves) and must return a sequence of objects BUMPS can
        evaluate through ``float()`` — typically ``bumps.parameter.Constraint``
        instances or any object whose ``__float__`` returns ``0`` when
        satisfied and the violation otherwise. The factory is invoked on
        every call because BUMPS parameters are rebuilt per fit: constraint
        operands *must* read these BUMPS parameters (the trial vector), not
        the EasyScience parameters, which are only updated inside the model
        evaluation and therefore lag — and freeze entirely while BUMPS skips
        the model in the infeasible region.

    Returns
    -------
    tuple[FitProblem, EvalCounter, Curve]
        The assembled problem, the evaluation counter wrapping the fit
        function (exposes ``count`` for evaluation bookkeeping), and the
        ``Curve`` model itself. The ``Curve`` is surfaced directly
        because ``FitProblem.fitness`` is deprecated in BUMPS (>= 1.0.4
        it emits a ``UserWarning``) — callers must not go through it.
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
    constraints = None
    if constraints_factory is not None:
        if not callable(constraints_factory):
            raise TypeError('constraints_factory must be callable')
        constraints = list(constraints_factory(dict(bumps_pars)))
    if constraints:
        return FitProblem(curve, constraints=constraints), fit_func, curve
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


def infeasible_constraints(problem: FitProblem, point: np.ndarray | None = None) -> list[str]:
    """Return the names of the problem's inequality constraints violated at ``point``.

    Parameters
    ----------
    problem : FitProblem
        A BUMPS problem built by :func:`build_curve_problem`.
    point : np.ndarray | None
        Parameter vector to test; when ``None`` the problem's current
        values are used. The problem's parameters are restored afterwards.

    Returns
    -------
    list[str]
        ``str(constraint)`` of each failing constraint; empty when the point
        is feasible or the problem carries no constraints. Used to flag
        progress payloads while BUMPS is on the penalty plateau, where the
        reported chi-squared is dominated by ``penalty_nllf`` and meaningless.
    """
    constraints = getattr(problem, 'constraints', None)
    # BUMPS stores the constraints as a plain list; anything else (e.g. a test
    # double) has no inequality constraints to evaluate.
    if not isinstance(constraints, (list, tuple)) or not constraints:
        return []
    if point is None:
        _, failing = problem.constraints_nllf()
        return [str(item) for item in failing]
    current = problem.getp()
    try:
        problem.setp(np.asarray(point, dtype=float))
        _, failing = problem.constraints_nllf()
    finally:
        problem.setp(current)
    return [str(item) for item in failing]
