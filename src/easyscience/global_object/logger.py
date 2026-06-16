# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import logging
import os
from collections.abc import Iterator

PACKAGE_LOGGER_NAME = 'easyscience'
_LOG_LEVEL_ENV_VAR = 'EASYSCIENCE_LOG_LEVEL'

_LEVEL_NAME_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}


def _resolve_log_level(
    raw: str | None,
    default: int = logging.WARNING,
) -> int:
    """Parse an environment-variable string into a logging level."""
    if raw is None:
        return default
    stripped = raw.strip()
    if stripped.isdigit():
        return int(stripped)
    upper = stripped.upper()
    if upper in _LEVEL_NAME_MAP:
        return _LEVEL_NAME_MAP[upper]
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
        self._effective_default = _resolve_log_level(
            os.environ.get(_LOG_LEVEL_ENV_VAR), default=log_level
        )
        self.logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        self.logger.setLevel(self._effective_default)
        self.level = self._effective_default

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
        """Log an ERROR-level message with traceback on the package-root logger."""
        self.logger.exception(msg, *args, **kwargs)

    # -- public API -------------------------------------------------------

    def set_level(self, level: int | str) -> None:
        """
        Set the effective level of the package-root logger.

        Parameters
        ----------
        level : int | str
            Logging level — e.g. ``logging.WARNING``, ``'ERROR'``.
        """
        resolved_level = _resolve_log_level(level) if isinstance(level, str) else level
        self.level = resolved_level
        self.logger.setLevel(resolved_level)

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
    def at_level(self, level: int | str) -> Iterator[None]:
        """
        Context manager that temporarily sets the package-root logger
        to *level*, restoring the previous level on exit.

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

        Example
        -------
        Setting the log-level only within the context:

        ```python
        with global_object.log.at_level(logging.ERROR):
            fitter.fit(x, y, weights)  # no core messages below ERROR
        ```
        """
        previous = self.level
        resolved_level = _resolve_log_level(level) if isinstance(level, str) else level
        self.level = resolved_level
        self.logger.setLevel(resolved_level)
        try:
            yield
        finally:
            self.level = previous
            self.logger.setLevel(previous)

    # -- deprecated helpers (compatibility shims) -------------------------

    def suspend(self):
        """Suppress all core log output (set level to CRITICAL+1)."""
        self.logger.setLevel(logging.CRITICAL + 1)

    def resume(self):
        """Restore the previously configured log level."""
        self.logger.setLevel(self.level)
