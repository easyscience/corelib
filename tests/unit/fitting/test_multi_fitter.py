# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

import numpy as np
import pytest

from easyscience import ObjBase
from easyscience import Parameter
from easyscience.fitting.fitter import Fitter
from easyscience.fitting.multi_fitter import MultiFitter


class Line(ObjBase):
    m: Parameter
    c: Parameter

    def __init__(self, m_val: float, c_val: float):
        m = Parameter('m', m_val)
        c = Parameter('c', c_val)
        super().__init__('line', m=m, c=c)

    def __call__(self, x):
        return self.m.value * x + self.c.value


class TestMultiFitter:
    @pytest.fixture
    def multi_fitter(self, monkeypatch):
        monkeypatch.setattr(Fitter, '_update_minimizer', MagicMock())
        fit_object_1 = Line(1.0, 0.5)
        fit_object_2 = Line(2.0, 1.5)
        return MultiFitter([fit_object_1, fit_object_2], [fit_object_1, fit_object_2])

    def test_fit_progress_callback(self, multi_fitter: MultiFitter):
        # When
        multi_fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        multi_fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        multi_fitter._post_compute_reshaping = MagicMock(return_value='fit_result')
        multi_fitter._minimizer = MagicMock()
        multi_fitter._minimizer.fit = MagicMock(return_value='result')
        progress_callback = MagicMock()

        # Then
        result = multi_fitter.fit(
            ['x_1', 'x_2'],
            ['y_1', 'y_2'],
            ['weights_1', 'weights_2'],
            progress_callback=progress_callback,
        )

        # Expect
        assert result == 'fit_result'
        multi_fitter._minimizer.fit.assert_called_once_with(
            'x_fit',
            'y_new',
            weights='weights',
            tolerance=None,
            max_evaluations=None,
            progress_callback=progress_callback,
        )


# ===================================================================
# MultiFitter.sample() — Bayesian DREAM sampling
# ===================================================================


class TestMultiFitterSample:
    @pytest.fixture
    def multi_fitter(self, monkeypatch):
        monkeypatch.setattr(Fitter, '_update_minimizer', MagicMock())
        fit_object_1 = Line(1.0, 0.5)
        fit_object_2 = Line(2.0, 1.5)
        return MultiFitter([fit_object_1, fit_object_2], [fit_object_1, fit_object_2])

    def test_sample_basic(self, multi_fitter: MultiFitter):
        """Verify sample() calls the minimizer's sample() and returns its result."""
        import numpy as np

        multi_fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        multi_fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        multi_fitter._minimizer = MagicMock()
        multi_fitter._minimizer.package = 'bumps'
        expected_result = {
            'draws': np.array([[1.0]]),
            'param_names': ['a'],
            'state': 'stub',
            'logp': None,
        }
        multi_fitter._minimizer.sample = MagicMock(return_value=expected_result)

        result = multi_fitter.sample(
            x=[np.array([1.0]), np.array([2.0])],
            y=[np.array([0.1]), np.array([0.2])],
            weights=[np.array([1.0]), np.array([1.0])],
            samples=100,
            burn=20,
            thin=2,
            population=5,
        )

        assert result == expected_result
        multi_fitter._minimizer.sample.assert_called_once()
        call_kwargs = multi_fitter._minimizer.sample.call_args.kwargs
        assert call_kwargs['samples'] == 100
        assert call_kwargs['burn'] == 20
        assert call_kwargs['thin'] == 2
        assert call_kwargs['population'] == 5
        assert call_kwargs['progress_callback'] is None

    def test_sample_raises_if_not_bumps(self, multi_fitter: MultiFitter):
        """sample() should raise RuntimeError if minimizer is not BUMPS."""
        multi_fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        multi_fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        multi_fitter._minimizer = MagicMock()
        multi_fitter._minimizer.package = 'lmfit'  # Not bumps

        with pytest.raises(RuntimeError, match='Bayesian sampling requires a BUMPS minimizer'):
            multi_fitter.sample(
                x=[np.array([1.0])],
                y=[np.array([0.1])],
                weights=[np.array([1.0])],
            )

    def test_sample_with_progress_callback(self, multi_fitter: MultiFitter):
        """Progress callback should be forwarded to minimizer.sample()."""
        multi_fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        multi_fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        multi_fitter._minimizer = MagicMock()
        multi_fitter._minimizer.package = 'bumps'
        multi_fitter._minimizer.sample = MagicMock(
            return_value={'draws': [], 'param_names': [], 'state': None, 'logp': None}
        )

        progress_callback = MagicMock()

        multi_fitter.sample(
            x=[np.array([1.0])],
            y=[np.array([0.1])],
            weights=[np.array([1.0])],
            progress_callback=progress_callback,
        )

        kwargs = multi_fitter._minimizer.sample.call_args.kwargs
        assert kwargs['progress_callback'] is progress_callback

    def test_sample_population_alias(self, multi_fitter: MultiFitter):
        """chains parameter is aliased to population."""
        multi_fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        multi_fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        multi_fitter._minimizer = MagicMock()
        multi_fitter._minimizer.package = 'bumps'
        multi_fitter._minimizer.sample = MagicMock(
            return_value={'draws': [], 'param_names': [], 'state': None, 'logp': None}
        )

        multi_fitter.sample(
            x=[np.array([1.0])],
            y=[np.array([0.1])],
            weights=[np.array([1.0])],
            chains=7,  # Should be forwarded as population=7
        )

        kwargs = multi_fitter._minimizer.sample.call_args.kwargs
        assert kwargs['population'] == 7
        assert kwargs['chains'] is None

    def test_sample_conflicting_population_raises(self, multi_fitter: MultiFitter):
        multi_fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        multi_fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        multi_fitter._minimizer = MagicMock()
        multi_fitter._minimizer.package = 'bumps'

        with pytest.raises(ValueError, match='Conflicting population'):
            multi_fitter.sample(
                x=[np.array([1.0])],
                y=[np.array([0.1])],
                weights=[np.array([1.0])],
                chains=5,
                population=10,
            )

    def test_sample_restores_original_fit_function(self, multi_fitter: MultiFitter):
        """After sample() completes (even on error) the original fit_function is restored."""
        original_ff = multi_fitter.fit_function
        multi_fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        multi_fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        multi_fitter._minimizer = MagicMock()
        multi_fitter._minimizer.package = 'bumps'
        multi_fitter._minimizer.sample = MagicMock(side_effect=RuntimeError('boom'))

        with pytest.raises(RuntimeError):
            multi_fitter.sample(
                x=[np.array([1.0])],
                y=[np.array([0.1])],
                weights=[np.array([1.0])],
            )

        assert multi_fitter.fit_function is original_ff


# ===================================================================
# MultiFitter._post_compute_reshaping
# ===================================================================


class TestPostComputeReshaping:
    def test_splits_single_result_into_two(self):
        """Verify _post_compute_reshaping splits combined result by dataset."""
        import numpy as np

        from easyscience.fitting.minimizers import FitResults

        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]
        mf = MultiFitter(fit_objects, fit_objects)

        # Simulate a combined result
        combined = FitResults()
        combined.success = True
        combined.x = np.array([0, 1, 2])
        combined.y_obs = np.array([0.5, 1.0, 1.5])
        combined.y_calc = np.array([0.51, 0.99, 1.51])
        combined.y_err = np.array([0.01, 0.02, 0.03])
        combined.p = {'pm': 1.0, 'pc': 0.5}
        combined.p0 = {'pm': 0.0, 'pc': 0.0}
        combined.n_evaluations = 100
        combined.iterations = 50
        combined.message = 'success'
        combined.minimizer_engine = 'bumps'
        combined.engine_result = 'engine_result'

        # Set dependent_dims as _precompute_reshaping would
        mf._dependent_dims = [(2,), (1,)]

        x_input = [np.array([0.0, 1.0]), np.array([2.0])]
        y_input = [np.array([0.5, 1.0]), np.array([1.5])]

        results = mf._post_compute_reshaping(combined, x_input, y_input)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is True
        assert len(results[0].y_obs) == 2
        assert len(results[1].y_obs) == 1
        assert results[0].total_results is combined
        assert results[1].total_results is combined
        assert np.allclose(results[0].y_calc, [0.51, 0.99])

    def test_handles_single_dataset(self):
        import numpy as np

        from easyscience.fitting.minimizers import FitResults

        fit_objects = [Line(1.0, 0.5)]
        mf = MultiFitter(fit_objects, fit_objects)

        combined = FitResults()
        combined.success = True
        combined.y_obs = np.array([1.0, 2.0, 3.0])
        combined.y_calc = np.array([1.1, 2.1, 3.1])
        combined.y_err = np.array([0.1, 0.1, 0.1])
        combined.p = {'pm': 1.0}
        combined.p0 = {'pm': 0.0}
        combined.n_evaluations = 50
        combined.iterations = 25
        combined.message = 'ok'
        combined.minimizer_engine = 'bumps'
        combined.engine_result = 'er'

        mf._dependent_dims = [(3,)]
        x_input = [np.array([0.0, 1.0, 2.0])]
        y_input = [np.array([1.0, 2.0, 3.0])]

        results = mf._post_compute_reshaping(combined, x_input, y_input)

        assert len(results) == 1
        assert np.allclose(results[0].y_calc, [1.1, 2.1, 3.1])


# ===================================================================
# MultiFitter._precompute_reshaping with weights=None
# ===================================================================


class TestPrecomputeReshaping:
    def test_weights_all_none_returns_none(self):
        """When all weights are None, _precompute_reshaping should return None for weights."""
        import numpy as np

        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]
        mf = MultiFitter(fit_objects, fit_objects)

        x = [np.array([1.0, 2.0]), np.array([3.0])]
        y = [np.array([1.5, 2.5]), np.array([4.5])]
        weights = [None, None]

        x_fit, x_new, y_new, w_new, dims = MultiFitter._precompute_reshaping(
            x, y, weights, vectorized=False
        )

        assert w_new is None
        assert len(dims) == 2
