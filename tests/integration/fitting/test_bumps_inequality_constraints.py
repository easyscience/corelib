# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Inequality constraints injected into the BUMPS ``FitProblem`` through
``constraints_factory``.

The factory receives the BUMPS parameters of the freshly built problem and
returns objects BUMPS evaluates with ``float()``: ``0`` when satisfied, the
violation otherwise. Operands *must* read the BUMPS parameters — BUMPS
evaluates constraints before the model and skips the model while any
constraint fails, so the EasyScience parameter values are stale at that
point (see ``build_curve_problem``).
"""

import numpy as np
import pytest

from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.fitting import AvailableMinimizers
from easyscience.fitting import Fitter
from easyscience.fitting import Sampler
from easyscience.fitting.minimizers.bumps_utils import build_curve_problem
from easyscience.fitting.minimizers.bumps_utils import infeasible_constraints
from easyscience.fitting.minimizers.minimizer_bumps import Bumps
from easyscience.variable import Parameter

PREFIX = 'p'


class Line(ModelBase):
    def __init__(self):
        super().__init__()
        self._slope = Parameter('slope', 1.0, min=-10, max=10)
        self._intercept = Parameter('intercept', 0.5, min=-10, max=10)

    @property
    def slope(self) -> Parameter:
        return self._slope

    @property
    def intercept(self) -> Parameter:
        return self._intercept

    def __call__(self, x):
        return self.slope.value * x + self.intercept.value


class _Sum:
    """Reads the live BUMPS trial values; records every value it sees."""

    def __init__(self, bumps_pars, names, seen):
        self._pars = [bumps_pars[PREFIX + n] for n in names]
        self._seen = seen

    def __float__(self):
        v = float(sum(float(p.value) for p in self._pars))
        self._seen.append(v)
        return v


class _LessThan:
    def __init__(self, lhs, bound):
        self.lhs, self.bound = lhs, bound

    def __float__(self):
        v = float(self.lhs)
        return 0.0 if v < self.bound else v - self.bound

    def __str__(self):
        return f'sum < {self.bound}'


@pytest.fixture
def clear_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


@pytest.fixture
def problem_setup(clear_map):
    model = Line()
    x = np.linspace(0.0, 10.0, 50)
    y = 3.0 * x + 2.0  # unconstrained optimum: slope 3, intercept 2 (sum 5)
    weights = np.ones_like(x)
    seen = []

    def factory(bumps_pars):
        names = [model.slope.unique_name, model.intercept.unique_name]
        return [_LessThan(_Sum(bumps_pars, names, seen), 4.0)]

    return model, x, y, weights, factory, seen


@pytest.mark.parametrize('engine', [AvailableMinimizers.Bumps, AvailableMinimizers.Bumps_newton])
def test_bumps_fit_respects_inequality(problem_setup, engine):
    model, x, y, weights, factory, seen = problem_setup
    fitter = Fitter(model, model)
    fitter.switch_minimizer(engine)

    result = fitter.fit(x, y, weights=weights, constraints_factory=factory)

    assert result.success
    assert model.slope.value + model.intercept.value <= 4.0 + 1e-6
    # The optimum sits on the boundary, not at the unconstrained (5.0) solution.
    assert model.slope.value + model.intercept.value == pytest.approx(4.0, abs=1e-3)


def test_constraint_observes_trial_vector_not_stale_values(problem_setup):
    """Regression for the 'LiveValue' design flaw: the constraint must see the
    optimizer's trial points, which differ from the EasyScience values that
    are only refreshed inside the model call."""
    model, x, y, weights, factory, seen = problem_setup
    fitter = Fitter(model, model)
    fitter.switch_minimizer(AvailableMinimizers.Bumps)
    fitter.fit(x, y, weights=weights, constraints_factory=factory)

    assert len({round(v, 6) for v in seen}) > 5


def test_feasible_problem_unaffected(problem_setup):
    model, x, y, weights, _, _ = problem_setup

    def loose(bumps_pars):
        names = [model.slope.unique_name, model.intercept.unique_name]
        return [_LessThan(_Sum(bumps_pars, names, []), 100.0)]

    fitter = Fitter(model, model)
    fitter.switch_minimizer(AvailableMinimizers.Bumps)
    result = fitter.fit(x, y, weights=weights, constraints_factory=loose)
    assert result.success
    assert model.slope.value == pytest.approx(3.0, abs=1e-3)
    assert model.intercept.value == pytest.approx(2.0, abs=1e-3)


def test_fitter_rejects_constraints_for_non_bumps_engines(problem_setup):
    model, x, y, weights, factory, _ = problem_setup
    fitter = Fitter(model, model)
    fitter.switch_minimizer(AvailableMinimizers.LMFit)
    with pytest.raises(ValueError, match='require the BUMPS engine'):
        fitter.fit(x, y, weights=weights, constraints_factory=factory)
    # ``None`` is simply dropped for engines that do not know the keyword.
    assert fitter.fit(x, y, weights=weights, constraints_factory=None).success


def test_bumps_rejects_factory_with_caller_supplied_model(problem_setup):
    model, x, y, weights, factory, _ = problem_setup
    minimizer = Bumps(model, model, AvailableMinimizers.Bumps)
    with pytest.raises(ValueError, match='caller-supplied model'):
        minimizer.fit(x, y, weights, model=object(), constraints_factory=factory)


def test_build_curve_problem_attaches_constraints_and_skips_model(problem_setup):
    model, x, y, weights, factory, _ = problem_setup
    engine = Bumps(model, model, AvailableMinimizers.Bumps)
    problem, counter, _ = build_curve_problem(engine, x, y, weights, constraints_factory=factory)

    assert len(problem.constraints) == 1
    names = [p.name for p in problem._parameters]
    infeasible = [9.0 if 'slope' in n else 9.0 for n in names]  # sum 18 > 4
    assert infeasible_constraints(problem, np.asarray(infeasible)) == ['sum < 4.0']
    # Evaluating an infeasible point must not call the model.
    before = counter.count
    cost = problem.nllf(np.asarray(infeasible))
    assert counter.count == before
    assert cost >= problem.penalty_nllf
    # A feasible point evaluates the model and reports no failing constraint.
    feasible = np.asarray([1.0, 1.0])
    assert infeasible_constraints(problem, feasible) == []
    problem.nllf(feasible)
    assert counter.count == before + 1


def test_build_curve_problem_validates_factory(problem_setup):
    model, x, y, weights, _, _ = problem_setup
    engine = Bumps(model, model, AvailableMinimizers.Bumps)
    with pytest.raises(TypeError, match='callable'):
        build_curve_problem(engine, x, y, weights, constraints_factory='nope')


def test_progress_payload_flags_infeasible_points(problem_setup):
    model, x, y, weights, factory, _ = problem_setup
    payloads = []
    fitter = Fitter(model, model)
    fitter.switch_minimizer(AvailableMinimizers.Bumps)
    model.slope.value = 5.0
    model.intercept.value = 5.0  # start infeasible (sum 10 > 4)
    fitter.fit(
        x, y, weights=weights, constraints_factory=factory, progress_callback=payloads.append
    )

    assert payloads, 'expected progress payloads'
    assert all('infeasible' in p and 'failing_constraints' in p for p in payloads)
    assert any(p['infeasible'] for p in payloads)
    assert payloads[-1]['infeasible'] is False

    # Without constraints the payload keeps the engine-agnostic key set.
    plain = []
    fitter.fit(x, y, weights=weights, progress_callback=plain.append)
    assert plain and all('infeasible' not in p for p in plain)


def test_dream_sampler_carries_constraints(problem_setup):
    model, x, y, weights, factory, seen = problem_setup
    fitter = Fitter(model, model)
    fitter.switch_minimizer(AvailableMinimizers.Bumps)
    sampler = Sampler(fitter, x, y, weights=weights, constraints_factory=factory)

    results = sampler.sample(samples=200, burn=20, thin=1, population=4)
    draws = np.asarray(results.draws)
    names = list(results.param_names)
    sums = (
        draws[:, names.index(model.slope.unique_name)]
        + draws[:, names.index(model.intercept.unique_name)]
    )
    # The retained posterior draws are (overwhelmingly) inside the feasible region.
    assert np.mean(sums < 4.0 + 1e-6) > 0.95
    n_seen = len(seen)
    sampler.extend(additional_samples=100, thin=1)
    assert len(seen) > n_seen, 'extend() must keep evaluating the same constraints'
