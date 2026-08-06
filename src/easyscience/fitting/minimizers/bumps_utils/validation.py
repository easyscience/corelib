# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Input validation shared by the BUMPS minimizer and the DREAM sampler."""

from __future__ import annotations

import numpy as np


def validate_run_settings(samples: int, burn: int, thin: int) -> None:
    """Validate the DREAM run settings.

    Parameters
    ----------
    samples : int
        Number of raw samples to draw; must be a positive integer.
    burn : int
        Burn-in generations to discard; must be a non-negative integer.
    thin : int
        Thinning interval; must be a positive integer.

    Raises
    ------
    ValueError
        If any value is out of range or not an integer.
    """
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError('samples must be a positive integer.')
    if not isinstance(burn, int) or burn < 0:
        raise ValueError('burn must be a non-negative integer.')
    if not isinstance(thin, int) or thin < 1:
        raise ValueError('thin must be a positive integer.')


def validate_arrays(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    check_finite_xy: bool = True,
) -> None:
    """Validate the (x, y, weights) arrays for a BUMPS problem.

    Checks shape agreement between the three arrays, finiteness and
    strict positivity of the weights, and — when ``check_finite_xy`` is
    ``True`` — finiteness of x and y. Sampling passes ``True``; the
    classical fit path passes ``False`` to keep its historically more
    permissive behaviour.

    Parameters
    ----------
    x : np.ndarray
        Independent variable array.
    y : np.ndarray
        Dependent variable array.
    weights : np.ndarray
        Weight array (converted to ``dy = 1 / weights`` downstream).
    check_finite_xy : bool, default=True
        Also require x and y to be free of NaN/infinite values.

    Raises
    ------
    ValueError
        If the shapes disagree, the weights are non-finite or
        non-positive, or (with ``check_finite_xy``) x/y are non-finite.
    """
    if y.shape != x.shape:
        raise ValueError('x and y must have the same shape.')

    if check_finite_xy:
        if not np.isfinite(x).all():
            raise ValueError('x cannot contain NaN or infinite values.')
        if not np.isfinite(y).all():
            raise ValueError('y cannot contain NaN or infinite values.')

    if weights.shape != x.shape:
        raise ValueError('Weights must have the same shape as x and y.')

    if not np.isfinite(weights).all():
        raise ValueError('Weights cannot be NaN or infinite.')

    if (weights <= 0).any():
        raise ValueError('Weights must be strictly positive and non-zero.')
