# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the shared BUMPS input validation helpers."""

import numpy as np
import pytest

from easyscience.fitting.minimizers.bumps_utils import validate_arrays
from easyscience.fitting.minimizers.bumps_utils import validate_run_settings


class TestValidateRunSettings:
    @pytest.mark.parametrize(
        'kwargs, match',
        [
            ({'samples': 0}, 'samples must be a positive integer'),
            ({'samples': -1}, 'samples must be a positive integer'),
            ({'samples': 10.0}, 'samples must be a positive integer'),
            # bool is an int subclass; True must not sneak through as 1.
            ({'samples': True}, 'samples must be a positive integer'),
            ({'burn': -1}, 'burn must be a non-negative integer'),
            ({'burn': 1.5}, 'burn must be a non-negative integer'),
            ({'burn': True}, 'burn must be a non-negative integer'),
            ({'burn': False}, 'burn must be a non-negative integer'),
            ({'thin': 0}, 'thin must be a positive integer'),
            ({'thin': 2.0}, 'thin must be a positive integer'),
            ({'thin': True}, 'thin must be a positive integer'),
        ],
    )
    def test_invalid_settings_raise(self, kwargs, match):
        settings = {'samples': 10, 'burn': 0, 'thin': 1}
        settings.update(kwargs)
        with pytest.raises(ValueError, match=match):
            validate_run_settings(**settings)

    def test_valid_settings_pass(self):
        validate_run_settings(samples=1, burn=0, thin=1)
        validate_run_settings(samples=10000, burn=2000, thin=10)


class TestValidateArrays:
    @staticmethod
    def _data():
        return {
            'x': np.array([1.0, 2.0]),
            'y': np.array([0.1, 0.2]),
            'weights': np.array([1.0, 1.0]),
        }

    @pytest.mark.parametrize(
        'overrides, match',
        [
            ({'y': np.array([0.1])}, 'x and y must have the same shape'),
            ({'weights': np.array([1.0])}, 'Weights must have the same shape'),
            ({'weights': np.array([1.0, np.nan])}, 'Weights cannot be NaN'),
            ({'weights': np.array([1.0, np.inf])}, 'Weights cannot be NaN'),
            ({'weights': np.array([1.0, 0.0])}, 'Weights must be strictly positive'),
            ({'weights': np.array([1.0, -1.0])}, 'Weights must be strictly positive'),
        ],
    )
    @pytest.mark.parametrize('check_finite_xy', [True, False])
    def test_shared_checks_raise(self, overrides, match, check_finite_xy):
        """Shape and weight checks apply on both the fit and sampling paths."""
        data = self._data()
        data.update(overrides)
        with pytest.raises(ValueError, match=match):
            validate_arrays(**data, check_finite_xy=check_finite_xy)

    @pytest.mark.parametrize(
        'overrides, match',
        [
            ({'x': np.array([1.0, np.nan])}, 'x cannot contain NaN'),
            ({'x': np.array([1.0, np.inf])}, 'x cannot contain NaN'),
            ({'y': np.array([0.1, np.nan])}, 'y cannot contain NaN'),
            ({'y': np.array([0.1, np.inf])}, 'y cannot contain NaN'),
        ],
    )
    def test_finite_xy_checked_only_when_requested(self, overrides, match):
        """x/y finiteness is enforced for sampling but not for the classical
        fit path, preserving the fit path's historically permissive behaviour."""
        data = self._data()
        data.update(overrides)
        with pytest.raises(ValueError, match=match):
            validate_arrays(**data, check_finite_xy=True)
        validate_arrays(**data, check_finite_xy=False)  # must not raise

    def test_valid_arrays_pass(self):
        validate_arrays(**self._data(), check_finite_xy=True)
