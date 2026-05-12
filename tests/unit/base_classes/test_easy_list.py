# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import pytest

from easyscience import global_object
from easyscience.base_classes.easy_list import EasyList
from easyscience.base_classes.model_base import ModelBase
from easyscience.base_classes.new_base import NewBase
from easyscience.variable import DescriptorNumber
from easyscience.variable import Parameter


class Alpha(NewBase):
    """Concrete subclass of NewBase for testing."""

    def __init__(self, unique_name=None, display_name=None):
        super().__init__(unique_name=unique_name, display_name=display_name)


class Beta(NewBase):
    """Another concrete subclass of NewBase for testing."""

    def __init__(self, unique_name=None, display_name=None):
        super().__init__(unique_name=unique_name, display_name=display_name)


class MockModel(ModelBase):
    """A ModelBase subclass with a Parameter and a DescriptorNumber for testing get_all_variables."""

    def __init__(self, unique_name=None, display_name=None, temperature=25, volume=1.0):
        super().__init__(unique_name=unique_name, display_name=display_name)
        self._temperature = Parameter(name='temperature', value=temperature)
        self._volume = DescriptorNumber(name='volume', value=volume)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature.value = value

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume.value = value


class MockModelNested(ModelBase):
    """A ModelBase subclass that contains another ModelBase to test nested variable collection."""

    def __init__(self, unique_name=None, display_name=None, component=None, pressure=0):
        super().__init__(unique_name=unique_name, display_name=display_name)
        self._pressure = Parameter(name='pressure', value=pressure)
        self._component = component or MockModel(unique_name='inner', temperature=30, volume=2.0)

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        self._pressure.value = value

    @property
    def component(self):
        return self._component


class TestEasyList:
    @pytest.fixture(autouse=True)
    def clear(self):
        global_object.map._clear()

    @pytest.fixture
    def populated_list(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        a3 = Alpha(unique_name='a3')
        return EasyList(a1, a2, a3, protected_types=Alpha)

    # --- Init ---

    def test_empty_init(self):
        el = EasyList()
        assert len(el) == 0
        assert len(el._protected_types) == 1
        assert el._protected_types[0] == NewBase

    def test_init_with_items(self):
        a = Alpha(unique_name='a1')
        b = Alpha(unique_name='a2')
        el = EasyList(a, b)
        assert len(el) == 2
        assert el[0] is a
        assert el[1] is b

    def test_init_with_list_of_items(self):
        items = [Alpha(unique_name='a1'), Alpha(unique_name='a2')]
        el = EasyList(items)
        assert len(el) == 2

    def test_init_single_protected_type(self):
        el = EasyList(protected_types=Alpha)
        assert el._protected_types == [Alpha]

    def test_init_list_of_protected_types(self):
        el = EasyList(protected_types=[Alpha, Beta])
        assert el._protected_types == [Alpha, Beta]

    def test_init_invalid_protected_types_raises(self):
        with pytest.raises(TypeError, match='protected_types must be a NewBase subclass'):
            EasyList(protected_types=int)

    def test_init_invalid_protected_types_list_raises(self):
        with pytest.raises(TypeError, match='protected_types must be a NewBase subclass'):
            EasyList(protected_types=[int, str])

    def test_init_with_data_kwarg(self):
        """Test deserialization-style init via 'data' keyword."""
        items = [Alpha(unique_name='a1'), Alpha(unique_name='a2')]
        el = EasyList(data=items)
        assert len(el) == 2

    # --- Protected types ---

    def test_append_wrong_type_raises(self):
        el = EasyList(protected_types=Alpha)
        with pytest.raises(TypeError, match='Items must be one of'):
            el.append(Beta(unique_name='b1'))

    def test_append_correct_type_succeeds(self):
        el = EasyList(protected_types=Alpha)
        a = Alpha(unique_name='a1')
        el.append(a)
        assert len(el) == 1

    def test_append_non_newbase_raises(self):
        el = EasyList()
        with pytest.raises(TypeError, match='Items must be one of'):
            el.append('not a NewBase')

    def test_insert_wrong_type_raises(self):
        el = EasyList(protected_types=Alpha)
        with pytest.raises(TypeError, match='Items must be one of'):
            el.insert(0, Beta(unique_name='b1'))

    def test_setitem_wrong_type_raises(self):
        a = Alpha(unique_name='a1')
        el = EasyList(a, protected_types=Alpha)
        with pytest.raises(TypeError, match='Items must be one of'):
            el[0] = Beta(unique_name='b1')

    def test_multiple_protected_types(self):
        el = EasyList(protected_types=[Alpha, Beta])
        a = Alpha(unique_name='a1')
        b = Beta(unique_name='b1')
        el.append(a)
        el.append(b)
        assert len(el) == 2

    def test_init_rejects_wrong_type(self):
        b = Beta(unique_name='b1')
        with pytest.raises(TypeError, match='Items must be one of'):
            EasyList(b, protected_types=Alpha)

    # --- __getitem__ ---

    def test_getitem_int(self, populated_list):
        assert populated_list[0].unique_name == 'a1'
        assert populated_list[1].unique_name == 'a2'
        assert populated_list[-1].unique_name == 'a3'

    def test_getitem_int_out_of_range(self, populated_list):
        with pytest.raises(IndexError):
            populated_list[10]

    def test_getitem_slice(self, populated_list):
        sliced = populated_list[0:2]
        assert isinstance(sliced, EasyList)
        assert len(sliced) == 2
        assert sliced[0].unique_name == 'a1'
        assert sliced[1].unique_name == 'a2'

    def test_getitem_str_lookup(self, populated_list):
        item = populated_list['a2']
        assert item.unique_name == 'a2'

    def test_getitem_str_not_found(self, populated_list):
        with pytest.raises(KeyError, match='No item with unique name'):
            populated_list['nonexistent']

    def test_getitem_invalid_type(self, populated_list):
        with pytest.raises(TypeError, match='Index must be an int, slice, or str'):
            populated_list[3.14]

    # --- __setitem__ ---

    def test_setitem_int(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, protected_types=Alpha)
        el[0] = a2
        assert el[0].unique_name == 'a2'

    def test_setitem_slice(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        a3 = Alpha(unique_name='a3')
        a4 = Alpha(unique_name='a4')
        el = EasyList(a1, a2, protected_types=Alpha)
        el[0:2] = [a3, a4]
        assert el[0].unique_name == 'a3'
        assert el[1].unique_name == 'a4'

    def test_setitem_self_replacement_int(self):
        """e[0] = e[0] should work without warning."""
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, a2, protected_types=Alpha)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            el[0] = el[0]
            assert len(w) == 0
        assert el[0].unique_name == 'a1'
        assert len(el) == 2

    def test_setitem_self_replacement_slice(self):
        """e[0:2] = e[0:2] should work without warning."""
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, a2, protected_types=Alpha)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            el[0:2] = [el[0], el[1]]
            assert len(w) == 0
        assert el[0].unique_name == 'a1'
        assert el[1].unique_name == 'a2'

    def test_setitem_slice_partial_self_replacement(self):
        """e[0:2] = [e[0], new] should only warn about the new item."""
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        a3 = Alpha(unique_name='a3')
        el = EasyList(a1, a2, protected_types=Alpha)
        el[0:2] = [el[0], a3]
        assert el[0].unique_name == 'a1'
        assert el[1].unique_name == 'a3'

    def test_setitem_slice_length_mismatch_raises(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        a3 = Alpha(unique_name='a3')
        el = EasyList(a1, a2, protected_types=Alpha)
        with pytest.raises(
            ValueError,
            match='Length of new values must match the length of the slice being replaced',
        ):
            el[0:2] = [a3]  # Only one item provided for a slice of length 2

    def test_setitem_invalid_index_type(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        with pytest.raises(TypeError, match='Index must be an int or slice'):
            el[3.14] = Alpha(unique_name='a2')

    def test_setitem_slice_non_iterable_raises(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        with pytest.raises(TypeError, match='Value must be an iterable for slice assignment'):
            el[0:1] = Alpha(unique_name='a2')

    # --- __delitem__ ---

    def test_delitem_int(self, populated_list):
        assert len(populated_list) == 3
        del populated_list[0]
        assert len(populated_list) == 2
        assert populated_list[0].unique_name == 'a2'

    def test_delitem_slice(self, populated_list):
        assert len(populated_list) == 3
        del populated_list[0:2]
        assert len(populated_list) == 1
        assert populated_list[0].unique_name == 'a3'

    def test_delitem_str(self, populated_list):
        del populated_list['a2']
        assert len(populated_list) == 2
        assert populated_list[0].unique_name == 'a1'
        assert populated_list[1].unique_name == 'a3'

    def test_delitem_str_not_found(self, populated_list):
        with pytest.raises(KeyError, match='No item with unique name'):
            del populated_list['nonexistent']

    def test_delitem_invalid_type(self, populated_list):
        with pytest.raises(TypeError, match='Index must be an int, slice, or str'):
            del populated_list[3.14]

    # --- Uniqueness ---

    def test_append_duplicate_warns(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            el.append(a1)
            assert len(w) == 1
            assert 'already in EasyList' in str(w[0].message)
        assert len(el) == 1

    def test_insert_duplicate_warns(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            el.insert(0, a1)
            assert len(w) == 1
            assert 'already in EasyList' in str(w[0].message)
        assert len(el) == 1

    def test_setitem_duplicate_warns(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, a2, protected_types=Alpha)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            el[0] = a2  # a2 already at index 1
            assert len(w) == 1
            assert 'already in EasyList' in str(w[0].message)
        # Original value should remain unchanged
        assert el[0].unique_name == 'a1'

    def test_setitem_slice_duplicate_warns(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        a3 = Alpha(unique_name='a3')
        a4 = Alpha(unique_name='a4')
        el = EasyList(a1, a2, a3, protected_types=Alpha)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            el[0:3] = [a1, a3, a4]  # a3 already at index 2
            assert len(w) == 1
            assert 'already in EasyList' in str(w[0].message)
        assert el[0].unique_name == 'a1'
        assert el[1].unique_name == 'a2'  # a3 should not replace a2 because it's a duplicate
        assert el[2].unique_name == 'a4'

    # --- insert ---

    def test_insert_at_beginning(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, protected_types=Alpha)
        el.insert(0, a2)
        assert el[0].unique_name == 'a2'
        assert el[1].unique_name == 'a1'

    def test_insert_at_end(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, protected_types=Alpha)
        el.insert(100, a2)
        assert el[-1].unique_name == 'a2'

    def test_insert_non_int_index_raises(self):
        el = EasyList(protected_types=Alpha)
        with pytest.raises(TypeError, match='Index must be an integer'):
            el.insert('bad', Alpha(unique_name='a1'))

    # --- __contains__ ---

    def test_contains_by_object(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        assert a1 in el

    def test_not_contains_by_object(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, protected_types=Alpha)
        assert a2 not in el

    def test_contains_by_str(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        assert 'a1' in el

    def test_not_contains_by_str(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        assert 'nonexistent' not in el

    # --- reverse ---
    def test_reverse(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, a2, protected_types=Alpha)
        reversed_el = list(reversed(el))
        assert reversed_el[0].unique_name == 'a2'
        assert reversed_el[1].unique_name == 'a1'

    # --- index ---

    def test_index_by_object(self, populated_list):
        item = populated_list[1]
        assert populated_list.index(item) == 1

    def test_index_by_str(self, populated_list):
        assert populated_list.index('a2', 0, 3) == 1

    def test_index_str_not_found(self, populated_list):
        with pytest.raises(ValueError, match='is not in EasyList'):
            populated_list.index('nonexistent', 0, 3)

    def test_index_by_object_not_found(self, populated_list):
        other = Alpha(unique_name='other')
        with pytest.raises(ValueError):
            populated_list.index(other, 0, len(populated_list))

    def test_index_not_in_range(self, populated_list):
        with pytest.raises(ValueError):
            populated_list.index('a2', 2, 3)

    # --- pop ---

    def test_pop_default_last(self, populated_list):
        item = populated_list.pop()
        assert item.unique_name == 'a3'
        assert len(populated_list) == 2

    def test_pop_by_int(self, populated_list):
        item = populated_list.pop(0)
        assert item.unique_name == 'a1'
        assert len(populated_list) == 2

    def test_pop_by_str(self, populated_list):
        item = populated_list.pop('a2')
        assert item.unique_name == 'a2'
        assert len(populated_list) == 2

    def test_pop_str_not_found(self, populated_list):
        with pytest.raises(KeyError, match='No item with unique name'):
            populated_list.pop('nonexistent')

    def test_pop_invalid_type(self, populated_list):
        with pytest.raises(TypeError, match='Index must be an int or str'):
            populated_list.pop(3.14)

    # --- sort ---

    def test_sort(self):
        a3 = Alpha(unique_name='c')
        a1 = Alpha(unique_name='a')
        a2 = Alpha(unique_name='b')
        el = EasyList(a3, a1, a2, protected_types=Alpha)
        el.sort(key=lambda x: x.unique_name)
        assert el[0].unique_name == 'a'
        assert el[1].unique_name == 'b'
        assert el[2].unique_name == 'c'

    def test_sort_reverse(self):
        a1 = Alpha(unique_name='a')
        a2 = Alpha(unique_name='b')
        a3 = Alpha(unique_name='c')
        el = EasyList(a1, a2, a3, protected_types=Alpha)
        el.sort(key=lambda x: x.unique_name, reverse=True)
        assert el[0].unique_name == 'c'
        assert el[1].unique_name == 'b'
        assert el[2].unique_name == 'a'

    # --- __iter__ / __len__ ---

    def test_len(self):
        el = EasyList(protected_types=Alpha)
        assert len(el) == 0
        el.append(Alpha(unique_name='a1'))
        assert len(el) == 1

    # --- __repr__ ---

    def test_repr(self):
        el = EasyList(protected_types=Alpha)
        r = repr(el)
        assert 'EasyList' in r
        assert 'length 0' in r
        assert 'Alpha' in r

    # --- MutableSequence behavior ---
    # If we were to inherit from List instead of MutableSequence, we would have to implement all of these manually.

    def test_iter(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, a2, protected_types=Alpha)
        names = [item.unique_name for item in el]
        assert names == ['a1', 'a2']

    def test_extend(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(protected_types=Alpha)
        el.extend([a1, a2])
        assert len(el) == 2

    def test_remove(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, a2, protected_types=Alpha)
        el.remove(a1)
        assert len(el) == 1
        assert el[0].unique_name == 'a2'

    def test_iadd(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, protected_types=Alpha)
        el += [a2]
        assert len(el) == 2

    def test_iadd_easylist(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, protected_types=Alpha)
        el += EasyList(a2, protected_types=Alpha)
        assert len(el) == 2

    def test_count(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        assert el.count(a1) == 1

    def test_clear(self):
        a1 = Alpha(unique_name='a1')
        el = EasyList(a1, protected_types=Alpha)
        el.clear()
        assert len(el) == 0

    # --- Serialization ---

    def test_to_dict(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, a2, unique_name='my_list', protected_types=Alpha)
        d = el.to_dict()
        assert 'data' in d
        assert len(d['data']) == 2
        assert 'protected_types' in d
        assert d['protected_types'][0]['@class'] == 'Alpha'

    def test_to_dict_default_protected_type_not_serialized(self):
        a1 = NewBase(unique_name='nb1')
        el = EasyList(a1, unique_name='my_list')
        d = el.to_dict()
        assert 'protected_types' not in d

    def test_from_dict_round_trip(self):
        a1 = Alpha(unique_name='a1')
        a2 = Alpha(unique_name='a2')
        el = EasyList(a1, a2, unique_name='my_list', protected_types=Alpha)
        d = el.to_dict()
        # Clear the global map so deserialized objects can reuse the same unique names
        global_object.map._clear()
        el2 = EasyList.from_dict(d)
        assert len(el2) == 2
        assert el2[0].unique_name == 'a1'
        assert el2[1].unique_name == 'a2'
        assert d == el2.to_dict()  # The dicts should be the same after round trip

    # --- get_all_variables ---

    def test_get_all_variables_empty_list(self):
        """An empty EasyList should return an empty list of variables."""
        el = EasyList(protected_types=ModelBase)
        assert el.get_all_variables() == []

    def test_get_all_variables_no_modelbase_elements(self):
        """An EasyList with only plain NewBase elements (not ModelBase) should return an empty list."""
        a1 = Alpha(unique_name='a1')
        a2 = NewBase(unique_name='nb1')
        el = EasyList(a1, a2)
        assert el.get_all_variables() == []

    def test_get_all_variables_single_modelbase(self):
        """A single ModelBase element should return its Parameter and DescriptorNumber."""
        m1 = MockModel(unique_name='m1', temperature=10, volume=5.0)
        el = EasyList(m1, protected_types=ModelBase)
        vars = el.get_all_variables()
        assert len(vars) == 2
        names = {v.name for v in vars}
        assert 'temperature' in names
        assert 'volume' in names
        # Verify specific values
        temp_var = next(v for v in vars if v.name == 'temperature')
        assert temp_var.value == 10
        vol_var = next(v for v in vars if v.name == 'volume')
        assert vol_var.value == 5.0

    def test_get_all_variables_multiple_modelbase(self):
        """Multiple ModelBase elements should return all their combined variables."""
        m1 = MockModel(unique_name='m1', temperature=10, volume=5.0)
        m2 = MockModel(unique_name='m2', temperature=99, volume=1.5)
        el = EasyList(m1, m2, protected_types=ModelBase)
        vars = el.get_all_variables()
        assert len(vars) == 4
        names = {v.name for v in vars}
        assert names == {'temperature', 'volume'}

    def test_get_all_variables_mixed_elements(self):
        """Only ModelBase-derived elements contribute variables; plain NewBase elements are skipped."""
        m1 = MockModel(unique_name='m1', temperature=10, volume=5.0)
        a1 = Alpha(unique_name='a1')
        el = EasyList(m1, a1)
        vars = el.get_all_variables()
        assert len(vars) == 2
        names = {v.name for v in vars}
        assert names == {'temperature', 'volume'}

    def test_get_all_variables_nested_model(self):
        """A ModelBase containing another ModelBase should recursively collect all variables."""
        inner = MockModel(unique_name='inner', temperature=30, volume=2.0)
        parent = MockModelNested(unique_name='parent', component=inner, pressure=100)
        el = EasyList(parent, protected_types=ModelBase)
        vars = el.get_all_variables()
        # parent: pressure (Parameter), inner: temperature (Parameter), volume (DescriptorNumber)
        assert len(vars) == 3
        names = {v.name for v in vars}
        assert names == {'pressure', 'temperature', 'volume'}

    def test_get_all_variables_returns_descriptorbase_instances(self):
        """All returned items should be instances of DescriptorBase."""
        m1 = MockModel(unique_name='m1', temperature=10, volume=5.0)
        m2 = MockModel(unique_name='m2', temperature=99, volume=1.5)
        el = EasyList(m1, m2, protected_types=ModelBase)
        for v in el.get_all_variables():
            from easyscience.variable.descriptor_base import DescriptorBase

            assert isinstance(v, DescriptorBase)

    def test_get_all_variables_nested_easylist(self):
        """An EasyList containing another EasyList with mixed NewBase/ModelBase elements
        should collect variables from the inner EasyList's ModelBase items,
        skipping plain NewBase items."""
        inner_model = MockModel(unique_name='inner_m', temperature=50, volume=3.0)
        inner_plain = Alpha(unique_name='inner_a')
        inner_list = EasyList(inner_model, inner_plain)
        outer_model = MockModel(unique_name='outer_m', temperature=70, volume=4.0)
        outer_list = EasyList(inner_list, outer_model)

        # --- outer_list structure ---
        assert len(outer_list) == 2
        assert outer_list[0] is inner_list
        assert outer_list[1] is outer_model

        # --- inner_list structure ---
        assert len(inner_list) == 2
        assert inner_list[0] is inner_model
        assert inner_list[1] is inner_plain
        # inner_list own get_all_variables should only see inner_model (skip Alpha)
        inner_vars = inner_list.get_all_variables()
        assert len(inner_vars) == 2

        # --- outer_list.get_all_variables ---
        vars = outer_list.get_all_variables()
        # inner_model: temperature (50), volume (3.0); outer_model: temperature (70), volume (4.0)
        assert len(vars) == 4

        # Verify all returned items are DescriptorBase instances
        for v in vars:
            assert isinstance(v, DescriptorNumber)

        # Collect temperatures and volumes from both models
        temps = {v.value for v in vars if v.name == 'temperature'}
        vols = {v.value for v in vars if v.name == 'volume'}
        assert temps == {50, 70}
        assert vols == {3.0, 4.0}
