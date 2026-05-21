# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import collections.abc
import functools
import logging
from time import time
from typing import Any
from typing import Callable

from easyscience import global_object


class memoized:
    """
    Decorator.

    Caches a function's return value each time it is called. If called
    later with the same arguments, the cached value is returned (not
    reevaluated).
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if not isinstance(args, collections.abc.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up.
            return self.func(*args)
        if args in self.cache:
            return self.cache[args]
        value = self.func(*args)
        self.cache[args] = value
        return value

    def __repr__(self) -> str:
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


def counted(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Count how many times a function has been called.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to be counted.

    Returns
    -------
    Callable[..., Any]
        Wrapped function with a ``n_calls`` counter attribute.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        wrapped.n_calls += 1
        return func(*args, **kwargs)

    wrapped.n_calls = 0
    return wrapped


def time_it(func):
    """
    Times a function and reports the time either to the class' log or
    the base logger :param func: function to be timed :return: callable
    function with timer.
    """
    name = func.__module__ + '.' + func.__name__
    time_logger = global_object.log.getLogger('timer.' + name)

    @functools.wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            time_logger.debug(f'\033[1;34;49mExecution time: {end_ if end_ > 0 else 0} ms\033[0m')

    return _time_it


def deprecated(func):
    """
    This is a decorator which can be used to mark functions as
    deprecated.

    It will result in a warning being emitted when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        logging.getLogger('easyscience.deprecated').warning(
            'Call to deprecated function %s.', func.__name__,
            stacklevel=3,
        )
        return func(*args, **kwargs)

    return new_func
