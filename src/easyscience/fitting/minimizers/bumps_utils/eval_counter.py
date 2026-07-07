# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import functools
import inspect
from typing import Callable


class EvalCounter:
    """
    Wrap a callable so the number of invocations is recorded on
    ``count``.

    Used by the BUMPS minimizer to count objective-function evaluations
    for cross-backend consistency with LMFit (``nfev``) and DFO-LS
    (``nf``).
    """

    def __init__(self, fn: Callable):
        self._fn = fn
        self.count = 0
        self.__name__ = getattr(fn, '__name__', self.__class__.__name__)
        self.__signature__ = inspect.signature(fn)
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._fn(*args, **kwargs)
