# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for easyscience.legacy.collection_base.CollectionBase.

These tests cover methods not exercised by the existing
tests/unit/base_classes/test_collection_base.py suite.
"""

import warnings

import pytest

from easyscience import DescriptorNumber
from easyscience import Parameter
from easyscience import global_object
from easyscience.legacy.collection_base import CollectionBase


@pytest.fixture(autouse=True)
def _clear_map():
    global_object.map._clear()
    yield
    global_object.map._clear()


@pytest.fixture
def setup_pars():
    return {
        'name': 'test',
        'par1': Parameter('p1', 0.1, fixed=True),
        'des1': DescriptorNumber('d1', 0.1),
        'par2': Parameter('p2', 0.2),
        'des2': DescriptorNumber('d2', 0.2),
        'par3': Parameter('p3', 0.3),
    }


# ---------------------------------------------------------------------------
# Deprecation warning on instantiation
# ---------------------------------------------------------------------------

def test_instantiation_emits_deprecation_warning():
    """Instantiating CollectionBase should emit a DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        CollectionBase('test')
        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert 'deprecated' in str(deprecation_warnings[0].message).lower()


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------

def test_sort_ascending(setup_pars):
    """Sort by parameter value in ascending order."""
    name = setup_pars.pop('name')
    coll = CollectionBase(name, **setup_pars)
    coll.sort(mapping=lambda item: item.value, reverse=False)
    values = [item.value for item in coll]
    assert values == sorted(values)


def test_sort_descending(setup_pars):
    """Sort by parameter value in descending order."""
    name = setup_pars.pop('name')
    coll = CollectionBase(name, **setup_pars)
    coll.sort(mapping=lambda item: item.value, reverse=True)
    values = [item.value for item in coll]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# data property
# ---------------------------------------------------------------------------

def test_data_property(setup_pars):
    """The data property returns a tuple of stored items."""
    name = setup_pars.pop('name')
    coll = CollectionBase(name, **setup_pars)
    d = coll.data
    assert isinstance(d, tuple)
    assert len(d) == 5
    # Items should be the same objects
    for item_from_data, item_from_iter in zip(d, coll):
        assert item_from_data is item_from_iter


# ---------------------------------------------------------------------------
# __setitem__ with an EasyScience object (not a Number)
# ---------------------------------------------------------------------------

def test_setitem_with_easyscience_object(setup_pars):
    """Replace an item at an index with another EasyScience object."""
    name = setup_pars.pop('name')
    coll = CollectionBase(name, **setup_pars)

    n_before = len(coll)
    old_item = coll[0]
    new_item = Parameter('replacement', 99.0)

    coll[0] = new_item

    assert len(coll) == n_before
    assert coll[0].name == 'replacement'
    assert coll[0].value == 99.0
    # Old item should be removed from the graph
    assert old_item.unique_name not in global_object.map.get_edges(coll)


# ---------------------------------------------------------------------------
# __getitem__ with duplicate names returns a new CollectionBase
# ---------------------------------------------------------------------------

def test_getitem_duplicate_names_returns_collection(setup_pars):
    """When multiple items share the same name, __getitem__ returns a sub-collection."""
    name = setup_pars.pop('name')
    # Add two items with the same display name
    p1 = Parameter('same_name', 1.0)
    p2 = Parameter('same_name', 2.0)
    coll = CollectionBase(name, p1, p2)

    result = coll['same_name']
    assert isinstance(result, CollectionBase)
    assert len(result) == 2


def test_getitem_nonexistent_name_raises(setup_pars):
    """Looking up a nonexistent name raises IndexError."""
    name = setup_pars.pop('name')
    coll = CollectionBase(name, **setup_pars)

    with pytest.raises(IndexError, match='Given index does not exist'):
        _ = coll['nonexistent']


# ---------------------------------------------------------------------------
# insert method (called by MutableSequence.append, but test edge case)
# ---------------------------------------------------------------------------

def test_insert_at_specific_index(setup_pars):
    """Insert an item at a specific index (not just via append)."""
    name = setup_pars.pop('name')
    coll = CollectionBase(name, **setup_pars)

    n_before = len(coll)
    new_item = Parameter('inserted', 42.0)

    coll.insert(2, new_item)

    assert len(coll) == n_before + 1
    assert coll[2].name == 'inserted'
    assert coll[2].value == 42.0
