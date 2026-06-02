# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

import numpy as np
import pytest

import easyscience.fitting.fitter
from easyscience import AvailableMinimizers
from easyscience import Fitter


class TestFitter:
    @pytest.fixture
    def fitter(self, monkeypatch):
        monkeypatch.setattr(Fitter, '_update_minimizer', MagicMock())
        self.mock_fit_object = MagicMock()
        self.mock_fit_function = MagicMock()
        return Fitter(self.mock_fit_object, self.mock_fit_function)

    def test_constructor(self, fitter: Fitter):
        # When Then Expect
        assert fitter._fit_object == self.mock_fit_object
        assert fitter._fit_function == self.mock_fit_function
        assert fitter._dependent_dims is None
        assert fitter._enum_current_minimizer is None  # == AvailableMinimizers.LMFit_leastsq
        assert fitter._minimizer is None
        fitter._update_minimizer.assert_called_once_with(AvailableMinimizers.LMFit_leastsq)

    def test_make_model(self, fitter: Fitter):
        # When
        mock_minimizer = MagicMock()
        mock_minimizer.make_model = MagicMock(return_value='model')
        fitter._minimizer = mock_minimizer

        # Then
        model = fitter.make_model('pars')

        # Expect
        assert model == 'model'
        mock_minimizer.make_model.assert_called_once_with('pars')

    def test_evaluate(self, fitter: Fitter):
        # When
        mock_minimizer = MagicMock()
        mock_minimizer.evaluate = MagicMock(return_value='result')
        fitter._minimizer = mock_minimizer

        # Then
        result = fitter.evaluate('pars')

        # Expect
        assert result == 'result'
        mock_minimizer.evaluate.assert_called_once_with('pars')

    def test_convert_to_pars_obj(self, fitter: Fitter):
        # When
        mock_minimizer = MagicMock()
        mock_minimizer.convert_to_pars_obj = MagicMock(return_value='obj')
        fitter._minimizer = mock_minimizer

        # Then
        obj = fitter.convert_to_pars_obj('pars')

        # Expect
        assert obj == 'obj'
        mock_minimizer.convert_to_pars_obj.assert_called_once_with('pars')

    def test_initialize(self, fitter: Fitter):
        # When
        mock_fit_object = MagicMock()
        mock_fit_function = MagicMock()

        # Then
        fitter.initialize(mock_fit_object, mock_fit_function)

        # Expect
        assert fitter._fit_object == mock_fit_object
        assert fitter._fit_function == mock_fit_function
        fitter._update_minimizer.count(2)

    def test_create(self, fitter: Fitter, monkeypatch):
        # When
        fitter._update_minimizer = MagicMock()
        mock_string_to_enum = MagicMock(return_value=10)
        monkeypatch.setattr(easyscience.fitting.fitter, 'from_string_to_enum', mock_string_to_enum)

        # Then
        fitter.create('great-minimizer')

        # Expect
        mock_string_to_enum.assert_called_once_with('great-minimizer')
        fitter._update_minimizer.assert_called_once_with(10)

    def test_switch_minimizer(self, fitter: Fitter, monkeypatch):
        # When
        mock_minimizer = MagicMock()
        fitter._minimizer = mock_minimizer
        mock_string_to_enum = MagicMock(return_value=10)
        monkeypatch.setattr(easyscience.fitting.fitter, 'from_string_to_enum', mock_string_to_enum)

        # Then
        fitter.switch_minimizer('great-minimizer')

        # Expect
        fitter._update_minimizer.count(2)
        mock_string_to_enum.assert_called_once_with('great-minimizer')

    def test_update_minimizer(self, monkeypatch):
        # When
        mock_fit_object = MagicMock()
        mock_fit_function = MagicMock()

        mock_string_to_enum = MagicMock(return_value=10)
        mock_factory = MagicMock(return_value='minimizer')
        monkeypatch.setattr(easyscience.fitting.fitter, 'from_string_to_enum', mock_string_to_enum)
        monkeypatch.setattr(easyscience.fitting.fitter, 'factory', mock_factory)
        fitter = Fitter(mock_fit_object, mock_fit_function)

        # Then
        fitter._update_minimizer('great-minimizer')

        # Expect
        assert fitter._enum_current_minimizer == 'great-minimizer'
        assert fitter._minimizer == 'minimizer'

    def test_available_minimizers(self, fitter: Fitter):
        # When
        minimizers = fitter.available_minimizers

        # Then Expect
        assert minimizers == [
            'LMFit',
            'LMFit_leastsq',
            'LMFit_powell',
            'LMFit_cobyla',
            'LMFit_differential_evolution',
            'LMFit_scipy_least_squares',
            'Bumps',
            'Bumps_simplex',
            'Bumps_newton',
            'Bumps_lm',
            'DFO',
            'DFO_leastsq',
        ]

    def test_minimizer(self, fitter: Fitter):
        # When
        fitter._minimizer = 'minimizer'

        # Then
        minimizer = fitter.minimizer

        # Expect
        assert minimizer == 'minimizer'

    def test_fit_function(self, fitter: Fitter):
        # When Then
        fit_function = fitter.fit_function

        # Expect
        assert fit_function == self.mock_fit_function

    def test_set_fit_function(self, fitter: Fitter):
        # When
        fitter._enum_current_minimizer = 'current_minimizer'

        # Then
        fitter.fit_function = 'new-fit-function'

        # Expect
        assert fitter._fit_function == 'new-fit-function'
        fitter._update_minimizer.assert_called_with('current_minimizer')

    def test_fit_object(self, fitter: Fitter):
        # When Then
        fit_object = fitter.fit_object

        # Expect
        assert fit_object == self.mock_fit_object

    def test_set_fit_object(self, fitter: Fitter):
        # When
        fitter._enum_current_minimizer = 'current_minimizer'

        # Then
        fitter.fit_object = 'new-fit-object'

        # Expect
        assert fitter.fit_object == 'new-fit-object'
        fitter._update_minimizer.assert_called_with('current_minimizer')

    def test_fit(self, fitter: Fitter):
        # When
        fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        fitter._post_compute_reshaping = MagicMock(return_value='fit_result')
        fitter._minimizer = MagicMock()
        fitter._minimizer.fit = MagicMock(return_value='result')

        # Then
        result = fitter.fit('x', 'y', 'weights', 'vectorized')

        # Expect
        fitter._precompute_reshaping.assert_called_once_with('x', 'y', 'weights', 'vectorized')
        fitter._fit_function_wrapper.assert_called_once_with('x_new', flatten=True)
        fitter._post_compute_reshaping.assert_called_once_with('result', 'x', 'y')
        assert result == 'fit_result'
        assert fitter._dependent_dims == 'dims'
        assert fitter._fit_function == self.mock_fit_function

    def test_fit_progress_callback(self, fitter: Fitter):
        # When
        fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        fitter._post_compute_reshaping = MagicMock(return_value='fit_result')
        fitter._minimizer = MagicMock()
        fitter._minimizer.fit = MagicMock(return_value='result')
        progress_callback = MagicMock()

        # Then
        result = fitter.fit('x', 'y', 'weights', 'vectorized', progress_callback=progress_callback)

        # Expect
        assert result == 'fit_result'
        fitter._minimizer.fit.assert_called_once_with(
            'x_fit',
            'y_new',
            weights='weights',
            tolerance=None,
            max_evaluations=None,
            progress_callback=progress_callback,
        )

    def test_post_compute_reshaping(self, fitter: Fitter):
        # When
        fit_result = MagicMock()
        fit_result.y_calc = np.array([[10], [20], [30]])
        fit_result.y_err = np.array([[40], [50], [60]])
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        # Then
        result = fitter._post_compute_reshaping(fit_result, x, y)

        # Expect
        assert np.array_equal(result.y_calc, np.array([10, 20, 30]))
        assert np.array_equal(result.y_err, np.array([40, 50, 60]))
        assert np.array_equal(result.x, x)
        assert np.array_equal(result.y_obs, y)


# TODO
#       def test_fit_function_wrapper()
#       def test_precompute_reshaping()


# ===================================================================
# Fitter.mcmc_sample() — Bayesian DREAM sampling
# ===================================================================


class TestFitterMcmcSample:
    @pytest.fixture
    def fitter(self, monkeypatch):
        monkeypatch.setattr(Fitter, '_update_minimizer', MagicMock())
        return Fitter(MagicMock(), MagicMock())

    def test_basic(self, fitter: Fitter):
        """mcmc_sample() calls minimizer.mcmc_sample() and returns its result."""
        fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'w_new', 'dims')
        )
        fitter._fit_function_wrapper = MagicMock(return_value='wrapped')
        fitter._minimizer = MagicMock()
        fitter._minimizer.package = 'bumps'
        expected = {
            'draws': np.array([[1.0]]),
            'param_names': ['a'],
            'internal_bumps_object': 'stub',
            'logp': None,
        }
        fitter._minimizer.mcmc_sample = MagicMock(return_value=expected)

        result = fitter.mcmc_sample(
            x=np.array([1.0]),
            y=np.array([0.1]),
            weights=np.array([1.0]),
            samples=100,
            burn=20,
            thin=2,
            population=5,
        )

        assert result == expected
        fitter._precompute_reshaping.assert_called_once_with(
            np.array([1.0]), np.array([0.1]), np.array([1.0]), False
        )
        fitter._fit_function_wrapper.assert_called_once_with('x_new', flatten=True)
        fitter._minimizer.mcmc_sample.assert_called_once()
        kw = fitter._minimizer.mcmc_sample.call_args.kwargs
        assert kw['x'] == 'x_fit'
        assert kw['y'] == 'y_new'
        assert kw['weights'] == 'w_new'
        assert kw['samples'] == 100
        assert kw['burn'] == 20
        assert kw['thin'] == 2
        assert kw['population'] == 5
        assert kw['progress_callback'] is None
        assert fitter._dependent_dims == 'dims'

    def test_raises_if_not_bumps(self, fitter: Fitter):
        """RuntimeError raised when the active minimizer is not BUMPS."""
        fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'w_new', 'dims')
        )
        fitter._fit_function_wrapper = MagicMock(return_value='wrapped')
        fitter._minimizer = MagicMock()
        fitter._minimizer.package = 'lmfit'

        with pytest.raises(RuntimeError, match='Bayesian sampling requires a BUMPS minimizer'):
            fitter.mcmc_sample(x=np.array([1.0]), y=np.array([0.1]), weights=np.array([1.0]))

    def test_progress_callback_forwarded(self, fitter: Fitter):
        """progress_callback is forwarded to minimizer.mcmc_sample()."""
        fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'w_new', 'dims')
        )
        fitter._fit_function_wrapper = MagicMock(return_value='wrapped')
        fitter._minimizer = MagicMock()
        fitter._minimizer.package = 'bumps'
        fitter._minimizer.mcmc_sample = MagicMock(
            return_value={
                'draws': [],
                'param_names': [],
                'internal_bumps_object': None,
                'logp': None,
            }
        )
        cb = MagicMock()

        fitter.mcmc_sample(
            x=np.array([1.0]),
            y=np.array([0.1]),
            weights=np.array([1.0]),
            progress_callback=cb,
        )

        assert fitter._minimizer.mcmc_sample.call_args.kwargs['progress_callback'] is cb

    def test_fit_function_restored_on_success(self, fitter: Fitter):
        """Original fit function is restored after a successful call."""
        fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'w_new', 'dims')
        )
        fitter._fit_function_wrapper = MagicMock(return_value='wrapped')
        fitter._minimizer = MagicMock()
        fitter._minimizer.package = 'bumps'
        fitter._minimizer.mcmc_sample = MagicMock(
            return_value={
                'draws': [],
                'param_names': [],
                'internal_bumps_object': None,
                'logp': None,
            }
        )
        original = fitter._fit_function

        fitter.mcmc_sample(x=np.array([1.0]), y=np.array([0.1]), weights=np.array([1.0]))

        assert fitter._fit_function is original

    def test_fit_function_restored_on_error(self, fitter: Fitter):
        """Original fit function is restored even when minimizer raises."""
        fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'w_new', 'dims')
        )
        fitter._fit_function_wrapper = MagicMock(return_value='wrapped')
        fitter._minimizer = MagicMock()
        fitter._minimizer.package = 'bumps'
        fitter._minimizer.mcmc_sample = MagicMock(side_effect=RuntimeError('boom'))
        original = fitter._fit_function

        with pytest.raises(RuntimeError):
            fitter.mcmc_sample(x=np.array([1.0]), y=np.array([0.1]), weights=np.array([1.0]))

        assert fitter._fit_function is original

    @pytest.mark.parametrize(
        'kwargs, match',
        [
            ({'samples': 0}, 'samples must be a positive integer'),
            ({'samples': -1}, 'samples must be a positive integer'),
            ({'burn': -1}, 'burn must be a non-negative integer'),
            ({'thin': 0}, 'thin must be a positive integer'),
        ],
    )
    def test_invalid_args_raise(self, fitter: Fitter, kwargs, match):
        """Invalid samples/burn/thin values raise ValueError before any I/O."""
        with pytest.raises(ValueError, match=match):
            fitter.mcmc_sample(
                x=np.array([1.0]),
                y=np.array([0.1]),
                weights=np.array([1.0]),
                samples=kwargs.get('samples', 10),
                burn=kwargs.get('burn', 0),
                thin=kwargs.get('thin', 1),
            )
