# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging


class Logger:
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.level = log_level
        self.logger.setLevel(self.level)

    def getLogger(
        self, logger_name: str, color: str = '32', defaults: bool = True
    ) -> logging.Logger:
        """
        Create a logger :param color:.

        Parameters
        ----------
        logger_name : str
            Logger name. Usually __name__ on creation.
        color : str, default='32'
            By default, '32'.
        defaults : bool, default=True
            Do you want to associate any current file loggers with this
            logger. By default, True.

        Returns
        -------
        logging.Logger
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
