# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging


class Logger:
    def __init__(self, log_level: int = logging.INFO):
        """Init function."""
        self.logger = logging.getLogger(__name__)
        self.level = log_level
        self.logger.setLevel(self.level)

    def getLogger(self, logger_name, color: str = '32', defaults: bool = True) -> logging:
        """Create a logger :param color:.

        Parameters
        ----------
        color : str, optional
            By default, '32'.
        logger_name :
            Logger name. Usually __name__ on creation.
        defaults : bool, optional
            Do you want to associate any current file
            loggers with this logger. By default, True.

        Returns
        -------
        logging
            A logger.
        """
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        # self.applyLevel(logger)
        # for handler_type in self._handlers:
        #     for handler in self._handlers[handler_type]:
        #         if handler_type == 'sys' or defaults:
        #             handler.formatter._fmt = self._makeColorText(color)
        #             logger.addHandler(handler)
        # logger.propagate = False
        # self._loggers.append(logger)
        return logger
