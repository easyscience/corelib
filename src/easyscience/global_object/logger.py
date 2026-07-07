# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import logging
import os
from collections.abc import Generator

PACKAGE_LOGGER_NAME = 'easyscience'
_LOG_LEVEL_ENV_VAR = 'EASYSCIENCE_LOG_LEVEL'

_LEVEL_NAME_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


def _parse_log_level(level: int | str) -> int:
    """
    Convert a level number or name into a numeric level, raising on an
    unknown name.
    """
    if isinstance(level, str):
        stripped = level.strip()
        if stripped.isdigit():
            return int(stripped)
        upper = stripped.upper()
        if upper in _LEVEL_NAME_MAP:
            return _LEVEL_NAME_MAP[upper]
        valid = ', '.join(sorted(_LEVEL_NAME_MAP))
        raise ValueError(f'Unknown log level {level!r}. Expected an integer or one of: {valid}.')
    # bool is a subclass of int, so reject it explicitly before accepting ints.
    if isinstance(level, bool) or not isinstance(level, int):
        raise TypeError(f'Log level must be int or str, not {type(level).__name__}')
    return level


def _env_log_level(raw: str | None, default: int) -> int:
    """
    Parse the EASYSCIENCE_LOG_LEVEL env var, falling back to *default*
    on an unset/bad value.
    """
    if raw is None:
        return default
    try:
        return _parse_log_level(raw)
    except ValueError:
        return default


class Logger:
    """
    Central logging controller for EasyScience.

    Owns the package-root logger ``easyscience`` and provides a
    convenience API to set its level. It never calls
    ``logging.basicConfig`` nor attaches a handler, so the host
    application keeps full control over where output goes.
    """

    def __init__(self, log_level: int = logging.WARNING):
        """
        Parameters
        ----------
        log_level : int, default=logging.WARNING
            Default level for the package-root logger. Overridden by the
            ``EASYSCIENCE_LOG_LEVEL`` environment variable when set.
        """
        effective_default = _env_log_level(os.environ.get(_LOG_LEVEL_ENV_VAR), default=log_level)
        self.logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        self.logger.setLevel(effective_default)
        self._level_before_suspend = effective_default

    # -- convenience delegation methods (mirror logging module) ------------

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a DEBUG-level message on the package-root logger."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an INFO-level message on the package-root logger."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a WARNING-level message on the package-root logger."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an ERROR-level message on the package-root logger."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a CRITICAL-level message on the package-root logger."""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """
        Log an ERROR-level message with traceback on the package-root
        logger.
        """
        self.logger.exception(msg, *args, **kwargs)

    # -- public API -------------------------------------------------------

    @property
    def level(self) -> int:
        """
        Get the effective level of the package-root logger.

        Returns
        -------
        int
            Numeric logging level currently set on the ``easyscience``
            package-root logger.
        """
        return self.logger.level

    @level.setter
    def level(self, value: int | str) -> None:
        """
        Set the level of the package-root logger.

        Parameters
        ----------
        value : int | str
            Logging level — a number (e.g. ``logging.WARNING``) or a
            level name (e.g. ``'ERROR'``). A string that is not a
            recognised level name or numeric string raises
            ``ValueError``.
        """
        self.logger.setLevel(_parse_log_level(value))

    def getLogger(self, logger_name: str) -> logging.Logger:
        """
        Create or retrieve a child logger under *easyscience*.

        The returned logger is left at ``logging.NOTSET`` so it inherits
        level and handler configuration from the package-root logger.

        Parameters
        ----------
        logger_name : str
            Logger name. Usually ``__name__`` of the calling module.

        Returns
        -------
        logging.Logger
            A child logger whose name is ``easyscience.<logger_name>``
            when *logger_name* does not already start with *easyscience*
            or a dot.
        """
        if not logger_name.startswith(PACKAGE_LOGGER_NAME) and not logger_name.startswith('.'):
            logger_name = f'{PACKAGE_LOGGER_NAME}.{logger_name}'
        return logging.getLogger(logger_name)

    @contextlib.contextmanager
    def at_level(self, level: int | str) -> Generator[None, None, None]:
        """
        Context manager that temporarily sets the package-root logger to
        *level*, restoring the previous level on exit.

        Parameters
        ----------
        level : int | str
            Temporary level for the package-root logger — e.g.
            ``logging.ERROR`` or ``'ERROR'``.

        Yields
        ------
        None
            Control is handed back to the ``with`` block with the
            temporary level applied.

        Examples
        --------
        Setting the log-level only within the context:

        ```python
        with global_object.log.at_level(logging.ERROR):
            fitter.fit(x, y, weights)  # no core messages below ERROR
        ```
        """
        previous = self.logger.level
        self.level = level
        try:
            yield
        finally:
            self.logger.setLevel(previous)

    # -- deprecated helpers (compatibility shims) -------------------------

    def suspend(self):
        """Suppress all core log output (set level to CRITICAL+1)."""
        self._level_before_suspend = self.logger.level
        self.logger.setLevel(logging.CRITICAL + 1)

    def resume(self):
        """
        Restore the level captured by the most recent suspend call.
        """
        self.level = self._level_before_suspend
