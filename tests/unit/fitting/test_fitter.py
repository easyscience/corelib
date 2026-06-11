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
