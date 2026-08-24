# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Round-trip of parameter dependencies through *nested* serialization.

``Parameter.as_dict`` has always emitted the dependency expression, but a
parameter nested inside another object used to be encoded by
``SerializerBase._recursive_encoder`` straight through ``_convert_to_dict``,
which bypassed ``as_dict`` — so every custom constraint was silently lost on
save/load. These tests pin the fixed behaviour end to end: encode via the
owning object, decode via the owning object, resolve, and check the dependent
parameter still tracks its driver.
"""

import json

import pytest

from easyscience import global_object
from easyscience.base_classes import EasyList
from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter
from easyscience.variable.parameter_dependency_resolver import (
    get_parameters_with_pending_dependencies,
)
from easyscience.variable.parameter_dependency_resolver import resolve_all_parameter_dependencies


class Pair(ModelBase):
    """Accepts floats or ready-made ``Parameter``s, like the library classes
    the serializer rebuilds when the object is nested inside a collection."""

    def __init__(self, first=1.0, second=2.0, **kwargs):
        super().__init__(**kwargs)
        self._first = (
            first
            if isinstance(first, Parameter)
            else Parameter('first', first, unit='m', min=0, max=100)
        )
        self._second = (
            second
            if isinstance(second, Parameter)
            else Parameter('second', second, unit='m', min=0, max=100)
        )

    @property
    def first(self) -> Parameter:
        return self._first

    @property
    def second(self) -> Parameter:
        return self._second


@pytest.fixture
def clear_global_map(monkeypatch):
    # The serializer only rebuilds classes it can import from an ``easy*``
    # module, so expose the test class through ``model_base`` for the test.
    import easyscience.base_classes.model_base as model_base_module

    monkeypatch.setattr(Pair, '__module__', model_base_module.__name__)
    monkeypatch.setattr(model_base_module, 'Pair', Pair, raising=False)
    global_object.map._clear()
    yield
    global_object.map._clear()


def _json_roundtrip(d: dict) -> dict:
    return json.loads(json.dumps(d))


def test_nested_parameter_dict_carries_dependency(clear_global_map):
    pair = Pair()
    pair.second.make_dependent_on('2 * a', {'a': pair.first})

    d = pair.to_dict()

    assert d['second']['_dependency_string'] == '2 * a'
    assert d['second']['_independent'] is False
    # The driver carries the id the dependent refers to.
    driver_id = d['first']['__serializer_id']
    assert d['second']['_dependency_map_serializer_ids'] == {'a': driver_id}


def test_nested_dependency_survives_model_roundtrip(clear_global_map):
    pair = Pair(first=3.0)
    pair.second.make_dependent_on('2 * a', {'a': pair.first})
    d = _json_roundtrip(pair.to_dict())
    global_object.map._clear()

    loaded = Pair.from_dict(d)
    assert len(get_parameters_with_pending_dependencies(loaded)) == 1
    resolve_all_parameter_dependencies(loaded)

    assert loaded.second.independent is False
    assert loaded.second.value == pytest.approx(6.0)
    loaded.first.value = 5.0
    assert loaded.second.value == pytest.approx(10.0)
    assert get_parameters_with_pending_dependencies(loaded) == []


def test_dependency_inside_easylist_is_found_by_resolver(clear_global_map):
    """Collections keep their items in a private ``_data`` attribute; the
    resolver must iterate them rather than skip them as private."""
    a = Pair(first=1.0, second=1.0)
    b = Pair(first=1.0, second=1.0)
    b.first.make_dependent_on('x * 2', {'x': a.first})
    items = EasyList(a, b)
    d = _json_roundtrip(items.to_dict())
    global_object.map._clear()

    loaded = EasyList.from_dict(d)
    resolve_all_parameter_dependencies(loaded)

    loaded[0].first.value = 4.0
    assert loaded[1].first.value == pytest.approx(8.0)


def test_old_files_without_dependency_keys_still_load(clear_global_map):
    pair = Pair()
    d = pair.to_dict()
    for key in ('second', 'first'):
        for old in (
            '_dependency_string',
            '_dependency_map_serializer_ids',
            '_independent',
            '__serializer_id',
        ):
            d[key].pop(old, None)
    global_object.map._clear()
    loaded = Pair.from_dict(d)
    assert loaded.first.independent and loaded.second.independent
    assert get_parameters_with_pending_dependencies(loaded) == []


def test_resolver_tolerates_cycles_in_object_graph(clear_global_map):
    class Node:
        def __init__(self):
            self.param = Parameter('p', 1.0)
            self.other = None

    x, y = Node(), Node()
    x.other, y.other = y, x  # back-references
    # Must terminate and find both parameters once each.
    resolve_all_parameter_dependencies([x, y])


def test_loose_descriptor_constant_in_dependency_survives_roundtrip(clear_global_map):
    """A plain DescriptorNumber used as a constant in an expression belongs to
    no object; it must be embedded with the dependent parameter."""
    from easyscience.variable import DescriptorNumber

    pair = Pair(first=30.0, second=1.0)
    total = DescriptorNumber('total', 120.0, unit='m')
    pair.second.make_dependent_on('total - a', {'total': total, 'a': pair.first})
    assert pair.second.value == pytest.approx(90.0)
    d = _json_roundtrip(pair.to_dict())
    assert 'total' in d['second']['_dependency_map_descriptors']
    global_object.map._clear()
    del total

    loaded = Pair.from_dict(d)
    resolve_all_parameter_dependencies(loaded)
    loaded.first.value = 50.0
    assert loaded.second.value == pytest.approx(70.0)
