# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from bumps.monitor import Monitor


class BumpsProgressMonitor(Monitor):
    """
    BUMPS :class:`Monitor` that forwards per-step progress information
    to a user-supplied callback.

    The monitor delegates payload construction to ``payload_builder`` so
    the BUMPS minimizer can keep all backend-specific payload semantics
    in one place.
    """

    def __init__(self, problem, callback, payload_builder):
        self._problem = problem
        self._callback = callback
        self._payload_builder = payload_builder

    def config_history(self, history):
        history.requires(step=1, point=1, value=1)

    def __call__(self, history):
        payload = self._payload_builder(
            problem=self._problem,
            iteration=int(history.step[0]),
            point=np.asarray(history.point[0]),
            nllf=float(history.value[0]),
        )
        self._callback(payload)
