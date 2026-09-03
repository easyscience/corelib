# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the shared BUMPS problem-construction helpers."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import easyscience.fitting.minimizers.bumps_utils.problem
from easyscience.fitting.minimizers.bumps_utils import build_curve_problem
from easyscience.fitting.minimizers.bumps_utils import parameter_names
from easyscience.fitting.minimizers.bumps_utils import parameter_snapshot
from easyscience.fitting.minimizers.bumps_utils import to_bumps_parameter


class TestToBumpsParameter:
    def test_convert_parameter_object(self) -> None:
        from easyscience.variable import Parameter

        param = Parameter('thickness', 42.0, min=0.0, max=100.0)
        param.fixed = False

        result = to_bumps_parameter(param)

        # to_bumps_parameter uses obj.unique_name which is auto-assigned
        assert result.name.startswith('p')
        assert result.value == 42.0
        assert result.bounds == (0.0, 100.0)
        assert result.fixed is False

    def test_convert_fixed_parameter(self) -> None:
        from easyscience.variable import Parameter

        param = Parameter('roughness', 5.0, min=0.0, max=20.0)
        param.fixed = True

        result = to_bumps_parameter(param)

        assert result.name.startswith('p')
        assert result.fixed is True


class TestBuildCurveProblem:
    """Curve/FitProblem assembly, with the BUMPS classes mocked out."""

    @pytest.fixture(autouse=True)
    def _mock_bumps_classes(self, monkeypatch):
        self.mock_curve_cls = MagicMock(return_value='curve')
        self.mock_problem_cls = MagicMock(return_value='problem')
        monkeypatch.setattr(
            easyscience.fitting.minimizers.bumps_utils.problem, 'Curve', self.mock_curve_cls
        )
        monkeypatch.setattr(
            easyscience.fitting.minimizers.bumps_utils.problem,
            'FitProblem',
            self.mock_problem_cls,
        )
        self.mock_convert = MagicMock(side_effect=lambda par: f'converted-{par.unique_name}')
        monkeypatch.setattr(
            easyscience.fitting.minimizers.bumps_utils.problem,
            'to_bumps_parameter',
            self.mock_convert,
        )

    @staticmethod
    def _engine_with_cached_pars(cached_pars):
        engine = MagicMock()
        engine._generate_fit_function = MagicMock(
            return_value=MagicMock(return_value=np.array([2.0]))
        )
        engine._cached_pars = cached_pars
        return engine

    def test_uses_cached_parameters_by_default(self):
        cached_par = MagicMock()
        cached_par.unique_name = 'alpha'
        engine = self._engine_with_cached_pars({'alpha': cached_par})

        problem, counter, curve = build_curve_problem(
            engine, np.array([1.0]), np.array([2.0]), np.array([4.0])
        )

        assert problem == 'problem'
        # The Curve is surfaced directly so callers never have to read the
        # deprecated ``FitProblem.fitness`` property (CR-1).
        assert curve == 'curve'
        engine._generate_fit_function.assert_called_once_with()
        self.mock_convert.assert_called_once_with(cached_par)
        assert self.mock_curve_cls.call_args.kwargs['palpha'] == 'converted-alpha'
        self.mock_problem_cls.assert_called_once_with('curve')

    def test_explicit_parameters_override_cache(self):
        engine = self._engine_with_cached_pars({'alpha': MagicMock(unique_name='alpha')})
        explicit = MagicMock()
        explicit.unique_name = 'beta'

        build_curve_problem(
            engine, np.array([1.0]), np.array([2.0]), np.array([4.0]), parameters=[explicit]
        )

        self.mock_convert.assert_called_once_with(explicit)
        assert 'pbeta' in self.mock_curve_cls.call_args.kwargs
        assert 'palpha' not in self.mock_curve_cls.call_args.kwargs

    def test_curve_receives_data_and_dy(self):
        """weights are converted to dy = 1 / weights."""
        engine = self._engine_with_cached_pars({})
        x = np.array([1.0, 2.0])
        y = np.array([10.0, 20.0])
        weights = np.array([2.0, 4.0])

        build_curve_problem(engine, x, y, weights)

        call = self.mock_curve_cls.call_args
        np.testing.assert_array_equal(call.args[1], x)
        np.testing.assert_array_equal(call.args[2], y)
        np.testing.assert_array_equal(call.kwargs['dy'], 1 / weights)

    def test_counter_wraps_fit_function(self):
        """The returned EvalCounter wraps the wrapped fit function and counts calls."""
        inner = MagicMock(return_value=np.array([11.0, 22.0]))
        engine = self._engine_with_cached_pars({})
        engine._generate_fit_function = MagicMock(return_value=inner)

        _, counter, _ = build_curve_problem(
            engine, np.array([1.0]), np.array([2.0]), np.array([4.0])
        )

        # The counter itself is what Curve receives as the fit function.
        assert self.mock_curve_cls.call_args.args[0] is counter
        assert counter.count == 0
        counter(np.array([1.0]))
        assert counter.count == 1
        inner.assert_called_once()


class TestParameterNames:
    def test_strips_prefix(self):
        params = []
        for name in ('palpha', 'pbeta'):
            p = MagicMock()
            p.name = name
            params.append(p)
        problem = MagicMock()
        problem._parameters = params

        assert parameter_names(problem) == ['alpha', 'beta']

    def test_tolerates_none_name(self):
        p = MagicMock()
        p.name = None
        problem = MagicMock()
        problem._parameters = [p]

        assert parameter_names(problem) == ['']


class TestParameterSnapshot:
    def test_snapshot_from_point(self) -> None:
        mock_problem = MagicMock()
        mock_problem.labels.return_value = ['palpha', 'pbeta']

        point = np.array([1.5, 2.5])

        snapshot = parameter_snapshot(mock_problem, point)

        assert snapshot == {'alpha': 1.5, 'beta': 2.5}
        mock_problem.getp.assert_not_called()

    def test_snapshot_falls_back_to_getp(self) -> None:
        mock_problem = MagicMock()
        mock_problem.labels.return_value = ['palpha']
        mock_problem.getp.return_value = np.array([3.5])

        snapshot = parameter_snapshot(mock_problem, None)

        assert snapshot == {'alpha': 3.5}
        mock_problem.getp.assert_called_once()
