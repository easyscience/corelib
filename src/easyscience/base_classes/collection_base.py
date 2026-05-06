# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""
.. deprecated::
    This module has been moved to `easyscience.legacy.collection_base`.
    Please update your imports.
"""

import warnings

from ..legacy.collection_base import CollectionBase  # noqa: F401

warnings.warn(
    'easyscience.base_classes.collection_base is deprecated. '
    'Please import from easyscience.legacy.collection_base instead.',
    DeprecationWarning,
    stacklevel=2,
)
