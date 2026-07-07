# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
.. deprecated::
    This module has been moved to ``easyscience.legacy.obj_base``.
    Please update your imports.
"""

from easyscience import global_object

from ..legacy.obj_base import ObjBase  # noqa: F401

global_object.log.warning(
    'easyscience.base_classes.obj_base is deprecated. '
    'Please import from easyscience.legacy.obj_base instead.',
    stacklevel=2,
)
