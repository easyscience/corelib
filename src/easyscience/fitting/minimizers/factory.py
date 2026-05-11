# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable

from .. import available_minimizers
from ..available_minimizers import AvailableMinimizers
from .minimizer_base import MinimizerBase

if available_minimizers.lmfit_engine_available:
    from .minimizer_lmfit import LMFit
if available_minimizers.dfo_engine_available:
    from .minimizer_dfo import DFO
if available_minimizers.bumps_engine_available:
    from .minimizer_bumps import Bumps


def factory(
    minimizer_enum: AvailableMinimizers, fit_object, fit_function: Callable
) -> MinimizerBase:
    if minimizer_enum.package == 'lm':
        minimizer = LMFit(obj=fit_object, fit_function=fit_function, minimizer_enum=minimizer_enum)

    elif minimizer_enum.package == 'bumps':
        minimizer = Bumps(obj=fit_object, fit_function=fit_function, minimizer_enum=minimizer_enum)

    elif minimizer_enum.package == 'dfo':
        minimizer = DFO(obj=fit_object, fit_function=fit_function, minimizer_enum=minimizer_enum)

    return minimizer
