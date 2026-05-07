# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
.. deprecated::
    This module has been moved to `easyscience.legacy.obj_base`.
    Please update your imports.
"""

import warnings

from ..legacy.obj_base import ObjBase  # noqa: F401

warnings.warn(
    'easyscience.base_classes.obj_base is deprecated. '
    'Please import from easyscience.legacy.obj_base instead.',
    DeprecationWarning,
    stacklevel=2,
)
