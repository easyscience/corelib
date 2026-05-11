# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import dataclass
from enum import Enum

# Change to importlib.metadata when Python 3.10 is the minimum version
# import importlib.metadata
# installed_packages = [x.name for x in importlib.metadata.distributions()]

lmfit_engine_available = False
try:
    import lmfit  # noqa: F401

    lmfit_engine_available = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn(
        'LMFit minimization is not available. Probably lmfit has not been installed.',
        ImportWarning,
        stacklevel=2,
    )

bumps_engine_available = False
try:
    import bumps  # noqa: F401

    bumps_engine_available = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn(
        'Bumps minimization is not available. Probably bumps has not been installed.',
        ImportWarning,
        stacklevel=2,
    )

dfo_engine_available = False
try:
    import dfols  # noqa: F401

    dfo_engine_available = True
except ImportError:
    # TODO make this a proper message (use logging?)
    warnings.warn(
        'DFO minimization is not available. Probably dfols has not been installed.',
        ImportWarning,
        stacklevel=2,
    )


@dataclass
class AvailableMinimizer:
    package: str
    method: str
    enum_id: int


class AvailableMinimizers(AvailableMinimizer, Enum):
    if lmfit_engine_available:
        LMFit = 'lm', 'leastsq', 11
        LMFit_leastsq = 'lm', 'leastsq', 12
        LMFit_powell = 'lm', 'powell', 13
        LMFit_cobyla = 'lm', 'cobyla', 14
        LMFit_differential_evolution = 'lm', 'differential_evolution', 15
        LMFit_scipy_least_squares = 'lm', 'least_squares', 16

    if bumps_engine_available:
        Bumps = 'bumps', 'amoeba', 21
        Bumps_simplex = 'bumps', 'amoeba', 22
        Bumps_newton = 'bumps', 'newton', 23
        Bumps_lm = 'bumps', 'lm', 24

    if dfo_engine_available:
        DFO = 'dfo', 'leastsq', 31
        DFO_leastsq = 'dfo', 'leastsq', 32


# Temporary solution to convert string to enum
def from_string_to_enum(minimizer_name: str) -> AvailableMinimizers:
    if minimizer_name == 'LMFit':
        minmizer_enum = AvailableMinimizers.LMFit
    elif minimizer_name == 'LMFit_leastsq':
        minmizer_enum = AvailableMinimizers.LMFit_leastsq
    elif minimizer_name == 'LMFit_powell':
        minmizer_enum = AvailableMinimizers.LMFit_powell
    elif minimizer_name == 'LMFit_cobyla':
        minmizer_enum = AvailableMinimizers.LMFit_cobyla
    elif minimizer_name == 'LMFit_differential_evolution':
        minmizer_enum = AvailableMinimizers.LMFit_differential_evolution
    elif minimizer_name == 'LMFit_scipy_least_squares':
        minmizer_enum = AvailableMinimizers.LMFit_scipy_least_squares

    elif minimizer_name == 'Bumps':
        minmizer_enum = AvailableMinimizers.Bumps
    elif minimizer_name == 'Bumps_simplex':
        minmizer_enum = AvailableMinimizers.Bumps_simplex
    elif minimizer_name == 'Bumps_newton':
        minmizer_enum = AvailableMinimizers.Bumps_newton
    elif minimizer_name == 'Bumps_lm':
        minmizer_enum = AvailableMinimizers.Bumps_lm

    elif minimizer_name == 'DFO':
        minmizer_enum = AvailableMinimizers.DFO
    elif minimizer_name == 'DFO_leastsq':
        minmizer_enum = AvailableMinimizers.DFO_leastsq
    else:
        raise ValueError(
            f'Invalid minimizer name: {minimizer_name}. The following minimizers are available: {[minimize.name for minimize in AvailableMinimizers]}'  # noqa: E501
        )

    return minmizer_enum
