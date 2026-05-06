# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the deprecated wrapper modules in easyscience.base_classes.

These modules now only emit DeprecationWarnings and re-export from
easyscience.legacy. We test that the warnings are raised and that
the classes are still usable.
"""

import warnings

import pytest

from easyscience import global_object


@pytest.fixture(autouse=True)
def _clear_map():
    """Clear the global object map before and after each test."""
    global_object.map._clear()
    yield
    global_object.map._clear()


# ---------------------------------------------------------------------------
# Deprecated wrapper: easyscience.base_classes.collection_base
# ---------------------------------------------------------------------------


def test_import_collection_base_warns():
    """Importing easyscience.base_classes.collection_base emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        from easyscience.base_classes.collection_base import CollectionBase  # noqa: F811

        assert len(w) >= 1, 'Expected at least one DeprecationWarning'
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, 'Expected a DeprecationWarning'
        msg = str(deprecation_warnings[0].message)
        assert 'deprecated' in msg.lower()
        assert 'legacy.collection_base' in msg


def test_collection_base_still_works_from_deprecated():
    """The class imported from the deprecated wrapper still works."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        from easyscience.base_classes.collection_base import CollectionBase  # noqa: F811

    from easyscience import Parameter

    p = Parameter('p1', 1.0)
    coll = CollectionBase('test', p)
    assert len(coll) == 1
    assert coll[0].name == 'p1'


# ---------------------------------------------------------------------------
# Deprecated wrapper: easyscience.base_classes.obj_base
# ---------------------------------------------------------------------------


def test_import_obj_base_warns():
    """Importing easyscience.base_classes.obj_base emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        from easyscience.base_classes.obj_base import ObjBase  # noqa: F811

        assert len(w) >= 1, 'Expected at least one DeprecationWarning'
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1, 'Expected a DeprecationWarning'
        msg = str(deprecation_warnings[0].message)
        assert 'deprecated' in msg.lower()
        assert 'legacy.obj_base' in msg


def test_obj_base_still_works_from_deprecated():
    """The class imported from the deprecated wrapper still works."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        from easyscience.base_classes.obj_base import ObjBase  # noqa: F811

    from easyscience import Parameter

    p = Parameter('p1', 1.0)
    obj = ObjBase('test', p1=p)
    assert obj.p1.value == 1.0
