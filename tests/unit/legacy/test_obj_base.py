# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for easyscience.legacy.obj_base.ObjBase.

These tests cover methods not exercised by the existing
tests/unit/base_classes/test_obj_base.py suite.
"""

import warnings
from typing import ClassVar

import pytest

from easyscience import Parameter
from easyscience import global_object
from easyscience.legacy.obj_base import ObjBase


@pytest.fixture(autouse=True)
def _clear_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


# ---------------------------------------------------------------------------
# Deprecation warning on instantiation
# ---------------------------------------------------------------------------

def test_instantiation_emits_deprecation_warning():
    """Instantiating ObjBase should emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        ObjBase('test')
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert 'deprecated' in str(deprecation_warnings[0].message).lower()


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

def test_repr():
    """Verify __repr__ returns the expected format."""
    obj = ObjBase('my_name')
    r = repr(obj)
    assert "ObjBase" in r
    assert "my_name" in r


# ---------------------------------------------------------------------------
# __setattr__ with interface bindings regeneration
# ---------------------------------------------------------------------------

def test_setattr_with_interface_calls_generate_bindings():
    """When replacing an attribute that has an interface, generate_bindings is called."""
    from unittest.mock import Mock

    class A(ObjBase):
        a: ClassVar[Parameter]

        def __init__(self, a: Parameter):
            super().__init__('A', a=a)

    p1 = Parameter('a', 1.0)
    a = A(p1)

    # Attach a mock interface
    mock_iface = Mock()
    a.interface = mock_iface

    # Replace the parameter — should trigger generate_bindings via __setattr__
    p2 = Parameter('a', 2.0)
    a.a = p2

    mock_iface.generate_bindings.assert_called()


# ---------------------------------------------------------------------------
# __setattr__ without annotation (the else branch)
# ---------------------------------------------------------------------------

def test_setattr_without_annotation():
    """Setting a non-annotated BasedBase/DescriptorBase attribute updates graph."""

    class A(ObjBase):
        def __init__(self, p: Parameter):
            super().__init__('A', p=p)

    p1 = Parameter('p', 1.0)
    a = A(p1)

    graph = global_object.map
    edges_before = set(graph.get_edges(a))

    # Replace the parameter with a new one
    p2 = Parameter('p', 2.0)
    a.p = p2

    edges_after = set(graph.get_edges(a))
    assert edges_before != edges_after
    assert p2.unique_name in edges_after
    assert p1.unique_name not in edges_after


# ---------------------------------------------------------------------------
# __setter with DescriptorBase path
# ---------------------------------------------------------------------------

def test_setter_sets_descriptor_value():
    """When setting a Descriptor via the logged property, the descriptor's value is updated."""
    from easyscience import DescriptorNumber

    d = DescriptorNumber('d1', 0.5)
    obj = ObjBase('test', d1=d)
    obj.d1 = 3.14
    assert obj.d1.value == 3.14
