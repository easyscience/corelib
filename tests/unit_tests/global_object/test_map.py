#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from easyscience.global_object.map import Map
from easyscience.Objects.variable.parameter import Parameter
from easyscience.Objects.ObjectClasses import BaseObj
import pytest
import gc
from easyscience import global_object

class TestMap:
    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    @pytest.fixture
    def base_object(self):
        return BaseObj(name="test")

    @pytest.fixture
    def parameter_object(self):
        return Parameter(value=2.0, name="test2")

    def test_add_vertex(self, clear, base_object, parameter_object):
        # When Then Expect
        assert len(global_object.map._store) == 2
        assert len(global_object.map._Map__type_dict) == 2

    def test_clear(self, clear, base_object):
        # When
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        # Then
        global_object.map._clear()
        # Expect
        assert len(global_object.map._store) == 0
        assert global_object.map._Map__type_dict == {}

    def test_weakref(self, clear):
        # When
        test_obj = BaseObj(name="test")
        assert len(global_object.map._store) == 1
        assert len(global_object.map._Map__type_dict) == 1
        # Then
        del test_obj
        gc.collect()
        # Expect
        assert len(global_object.map._store) == 0
        assert len(global_object.map._Map__type_dict) == 0

    def test_vertices(self, clear, base_object, parameter_object):
        # When Then Expect
        assert global_object.map.vertices() == [base_object.unique_name, parameter_object.unique_name]

    def test_get_item_by_key(self, clear, base_object, parameter_object):
        # When Then Expect
        assert global_object.map.get_item_by_key(base_object.unique_name) == base_object
        assert global_object.map.get_item_by_key(parameter_object.unique_name) == parameter_object

    @pytest.mark.parametrize("cls, kwargs", [(BaseObj, {}), (Parameter, {"value": 2.0})])
    def test_identical_unique_names_exception(self, clear, cls, kwargs):
        # When
        test_obj = cls(name="test", unique_name="test", **kwargs)
        # Then Expect
        with pytest.raises(ValueError):
            test_obj2 = cls(name="test2", unique_name="test", **kwargs)

    def test_unique_name_change_still_in_map(self, clear, base_object, parameter_object):
        # When
        assert global_object.map.get_item_by_key("BaseObj_0") == base_object
        assert global_object.map.get_item_by_key("Parameter_0") == parameter_object
        # Then
        base_object.unique_name = "test3"
        parameter_object.unique_name = "test4"
        # Expect
        assert global_object.map.get_item_by_key("BaseObj_0") == base_object
        assert global_object.map.get_item_by_key("Parameter_0") == parameter_object
        assert global_object.map.get_item_by_key("test3") == base_object
        assert global_object.map.get_item_by_key("test4") == parameter_object

