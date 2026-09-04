# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import json

import pytest

from easyscience import Parameter
from easyscience import global_object
from easyscience.variable.parameter_dependency_resolver import deserialize_and_resolve_parameters
from easyscience.variable.parameter_dependency_resolver import (
    get_parameters_with_pending_dependencies,
)
from easyscience.variable.parameter_dependency_resolver import resolve_all_parameter_dependencies


class TestParameterDependencySerialization:
    @pytest.fixture
    def clear_global_map(self):
        """This fixture pattern:
          - Clears the map before each test (clean slate)
          - Yields control to the test
          - Clears the map after each test (cleanup)

        Dependency serialization tests require more robust
        setup-yield-cleanup pattern because they involve complex
        object lifecycles with serialization, deserialization,
        and dependency resolution that are particularly sensitive
        to global state contamination.
        """
        # The global map uses weakref.WeakValueDictionary() for object storage,
        # but also maintains strong references in __type_dict that need explicit cleanup.
        global_object.map._clear()
        yield
        # final cleanup after test
        global_object.map._clear()

    def test_independent_parameter_serialization(self, clear_global_map):
        """Test that independent parameters serialize normally without dependency info."""
        param = Parameter(name='test', value=5.0, unit='m', min=0, max=10)

        # Serialize
        serialized = param.as_dict()

        # Should not contain dependency fields
        assert '_dependency_string' not in serialized
        assert '_dependency_map_serializer_ids' not in serialized
        assert '_independent' not in serialized

        # Deserialize
        global_object.map._clear()
        new_param = Parameter.from_dict(serialized)

        # Should be identical
        assert new_param.name == param.name
        assert new_param.value == param.value
        assert new_param.unit == param.unit
        assert new_param.independent is True

    def test_dependent_parameter_serialization(self, clear_global_map):
        """Test serialization of parameters with dependencies."""
        # Create independent parameter
        a = Parameter(name='a', value=2.0, unit='m', min=0, max=10)

        # Create dependent parameter
        b = Parameter.from_dependency(
            name='b', dependency_expression='2 * a', dependency_map={'a': a}, unit='m'
        )

        # Serialize dependent parameter
        serialized = b.as_dict()

        # Should contain dependency information
        assert serialized['_dependency_string'] == '2 * a'
        assert serialized['_dependency_map_serializer_ids'] == {
            'a': a._DescriptorNumber__serializer_id
        }
        assert serialized['_independent'] is False

        # Deserialize
        global_object.map._clear()
        new_b = Parameter.from_dict(serialized)

        # Should have pending dependency info
        assert hasattr(new_b, '_pending_dependency_string')
        assert new_b._pending_dependency_string == '2 * a'
        assert new_b._pending_dependency_map_serializer_ids == {
            'a': a._DescriptorNumber__serializer_id
        }
        assert new_b.independent is True  # Initially independent until dependencies resolved

    def test_dependency_resolution_after_deserialization(self, clear_global_map):
        """Test that dependencies are properly resolved after deserialization."""
        # Create test parameters with dependencies
        a = Parameter(name='a', value=2.0, unit='m', min=0, max=10)
        b = Parameter(name='b', value=3.0, unit='m', min=0, max=10)

        c = Parameter.from_dependency(
            name='c',
            dependency_expression='a + b',
            dependency_map={'a': a, 'b': b},
            unit='m',
        )

        # Verify original dependency works
        assert c.value == 5.0  # 2 + 3

        # Serialize all parameters
        params_data = {'a': a.as_dict(), 'b': b.as_dict(), 'c': c.as_dict()}

        # Clear and deserialize (manual approach)
        global_object.map._clear()
        new_params = {}
        for name, data in params_data.items():
            new_params[name] = Parameter.from_dict(data)

        # Before resolution, c should be independent with pending dependency
        assert new_params['c'].independent is True
        assert hasattr(new_params['c'], '_pending_dependency_string')

        # Resolve dependencies
        resolve_all_parameter_dependencies(new_params)

        # Alternative simplified approach using the helper function:
        # global_object.map._clear()
        # new_params = deserialize_and_resolve_parameters(params_data)

        # After resolution, c should be dependent and functional
        assert new_params['c'].independent is False
        assert new_params['c'].value == 5.0  # Still 2 + 3

        # Test that dependency still works
        new_params['a'].value = 10.0
        assert new_params['c'].value == 13.0  # 10 + 3

    def test_dependency_resolution_after_deserialization_desired_unit(self, clear_global_map):
        """Test that dependencies are properly resolved after deserialization."""
        # Create test parameters with dependencies
        a = Parameter(name='a', value=2.0, unit='m', min=0, max=10)
        b = Parameter(name='b', value=3.0, unit='m', min=0, max=10)

        c = Parameter.from_dependency(
            name='c',
            dependency_expression='a + b',
            dependency_map={'a': a, 'b': b},
            desired_unit='cm',
        )

        # Verify original dependency works
        assert c.value == 5.0 * 100  # 2 + 3
        assert c.unit == 'cm'

        # Serialize all parameters
        params_data = {'a': a.as_dict(), 'b': b.as_dict(), 'c': c.as_dict()}

        # Clear and deserialize (manual approach)
        global_object.map._clear()
        new_params = {}
        for name, data in params_data.items():
            new_params[name] = Parameter.from_dict(data)

        # Before resolution, c should be independent with pending dependency
        assert new_params['c'].independent is True
        assert hasattr(new_params['c'], '_pending_dependency_string')

        # Resolve dependencies
        resolve_all_parameter_dependencies(new_params)

        # Alternative simplified approach using the helper function:
        # global_object.map._clear()
        # new_params = deserialize_and_resolve_parameters(params_data)

        # After resolution, c should be dependent and functional
        assert new_params['c'].independent is False
        assert new_params['c']._desired_unit == 'cm'  # Desired unit should be preserved
        assert new_params['c'].value == 5.0 * 100  # Still 2 + 3, converted to cm

        # Test that dependency still works
        new_params['a'].value = 10.0
        assert new_params['c'].value == 13.0 * 100  # 10 + 3
        assert new_params['c'].unit == 'cm'

    def test_unique_name_dependency_serialization(self, clear_global_map):
        """Test serialization of dependencies using unique names."""
        a = Parameter(name='a', value=3.0, unit='m', min=0, max=10)

        # Create dependent parameter using unique name
        b = Parameter.from_dependency(
            name='b',
            dependency_expression='2 * "Parameter_0"',  # Using unique name
            unit='m',
        )

        # Serialize both parameters
        a_serialized = a.as_dict()
        b_serialized = b.as_dict()

        # Should contain unique name mapping
        assert b_serialized['_dependency_string'] == '2 * __Parameter_0__'
        assert '__Parameter_0__' in b_serialized['_dependency_map_serializer_ids']
        assert (
            b_serialized['_dependency_map_serializer_ids']['__Parameter_0__']
            == a._DescriptorNumber__serializer_id
        )

        # Deserialize both and resolve
        global_object.map._clear()
        c = Parameter(
            name='c', value=0.0
        )  # Dummy to occupy unique name, to force new unique_names

        # Remove unique_name from serialized data to force generation of new unique names
        a_serialized.pop('unique_name', None)
        b_serialized.pop('unique_name', None)

        new_b = Parameter.from_dict(b_serialized)
        new_a = Parameter.from_dict(a_serialized)
        resolve_all_parameter_dependencies({'a': new_a, 'b': new_b})

        # Should work correctly
        assert new_b.independent is False
        new_a.value = 4.0
        assert new_b.value == 8.0  # 2 * 4

    def test_json_serialization_roundtrip(self, clear_global_map):
        """Test that parameter dependencies survive JSON serialization."""
        # Create parameters with dependencies
        length = Parameter(name='length', value=10.0, unit='m', min=0, max=100)
        width = Parameter(name='width', value=5.0, unit='m', min=0, max=50)

        area = Parameter.from_dependency(
            name='area',
            dependency_expression='length * width',
            dependency_map={'length': length, 'width': width},
            unit='m^2',
        )

        # Serialize to JSON
        params_data = {
            'length': length.as_dict(),
            'width': width.as_dict(),
            'area': area.as_dict(),
        }
        json_str = json.dumps(params_data, default=str)

        # Deserialize from JSON
        global_object.map._clear()
        loaded_data = json.loads(json_str)
        new_params = {}
        for name, data in loaded_data.items():
            new_params[name] = Parameter.from_dict(data)

        # Resolve dependencies
        resolve_all_parameter_dependencies(new_params)

        # Test functionality
        assert new_params['area'].value == 50.0  # 10 * 5

        # Test dependency updates
        new_params['length'].value = 20.0
        assert new_params['area'].value == 100.0  # 20 * 5

    def test_multiple_dependent_parameters(self, clear_global_map):
        """Test serialization with multiple dependent parameters."""
        # Create a chain of dependencies
        x = Parameter(name='x', value=2.0, unit='m', min=0, max=10)

        y = Parameter.from_dependency(
            name='y', dependency_expression='2 * x', dependency_map={'x': x}, unit='m'
        )

        z = Parameter.from_dependency(
            name='z',
            dependency_expression='y + x',
            dependency_map={'y': y, 'x': x},
            unit='m',
        )

        # Verify original chain works
        assert y.value == 4.0  # 2 * 2
        assert z.value == 6.0  # 4 + 2

        # Serialize all
        params_data = {'x': x.as_dict(), 'y': y.as_dict(), 'z': z.as_dict()}

        # Deserialize and resolve
        global_object.map._clear()
        new_params = {}
        for name, data in params_data.items():
            new_params[name] = Parameter.from_dict(data)

        resolve_all_parameter_dependencies(new_params)

        # Test chain still works
        assert new_params['y'].value == 4.0
        assert new_params['z'].value == 6.0

        # Test cascade updates
        new_params['x'].value = 5.0
        assert new_params['y'].value == 10.0  # 2 * 5
        assert new_params['z'].value == 15.0  # 10 + 5

    def test_dependency_with_descriptor_number(self, clear_global_map):
        """Test that dependencies involving DescriptorNumber serialize correctly."""
        from easyscience.variable import DescriptorNumber

        # When

        x = DescriptorNumber(name='x', value=3.0, unit='m')
        y = Parameter(name='y', value=4.0, unit='m')
        z = Parameter.from_dependency(
            name='z',
            dependency_expression='x + y',
            dependency_map={'x': x, 'y': y},
        )

        # Verify original functionality
        assert z.value == 7.0  # 3 + 4

        # Then
        # Serialize all
        params_data = {'x': x.as_dict(), 'y': y.as_dict(), 'z': z.as_dict()}
        # Deserialize and resolve
        global_object.map._clear()
        new_params = {}
        for name, data in params_data.items():
            if name == 'x':
                new_params[name] = DescriptorNumber.from_dict(data)
            else:
                new_params[name] = Parameter.from_dict(data)

        resolve_all_parameter_dependencies(new_params)

        # Expect
        # Test that functionality still works
        assert new_params['z'].value == 7.0  # 3 + 4
        new_x = new_params['x']
        new_y = new_params['y']
        new_x.value = 4.0
        assert new_params['z'].value == 8.0  # 4 + 4
        new_y.value = 6.0
        assert new_params['z'].value == 10.0  # 4 + 6

    def test_get_parameters_with_pending_dependencies(self, clear_global_map):
        """Test utility function for finding parameters with pending dependencies."""
        # Create parameters
        a = Parameter(name='a', value=1.0, unit='m')
        b = Parameter.from_dependency(
            name='b', dependency_expression='2 * a', dependency_map={'a': a}, unit='m'
        )

        # Serialize and deserialize
        params_data = {'a': a.as_dict(), 'b': b.as_dict()}
        global_object.map._clear()
        new_params = {}
        for name, data in params_data.items():
            new_params[name] = Parameter.from_dict(data)

        # Find pending dependencies
        pending = get_parameters_with_pending_dependencies(new_params)

        assert len(pending) == 1
        assert pending[0].name == 'b'
        assert hasattr(pending[0], '_pending_dependency_string')

        # After resolution, should be empty
        resolve_all_parameter_dependencies(new_params)
        pending_after = get_parameters_with_pending_dependencies(new_params)
        assert len(pending_after) == 0

    def test_error_handling_missing_dependency(self, clear_global_map):
        """Test error handling when dependency cannot be resolved."""
        a = Parameter(name='a', value=1.0, unit='m')
        b = Parameter.from_dependency(
            name='b', dependency_expression='2 * a', dependency_map={'a': a}, unit='m'
        )

        # Serialize b but not a
        b_data = b.as_dict()

        # Deserialize without a in the global map
        global_object.map._clear()
        new_b = Parameter.from_dict(b_data)

        # Should raise error when trying to resolve
        with pytest.raises(ValueError, match='Cannot find parameter with serializer_id'):
            new_b.resolve_pending_dependencies()

    def test_backward_compatibility_base_deserializer(self, clear_global_map):
        """Test that the base deserializer path still works for dependent parameters."""
        from easyscience.io.serializer_dict import SerializerDict

        # Create dependent parameter
        a = Parameter(name='a', value=2.0, unit='m')
        b = Parameter.from_dependency(
            name='b', dependency_expression='3 * a', dependency_map={'a': a}, unit='m'
        )

        # Use base serializer path (SerializerDict.decode)
        serialized = SerializerDict().encode(b)
        global_object.map._clear()

        # This should not raise the "_independent" error anymore
        deserialized = SerializerDict.decode(serialized)

        # Should be a valid Parameter (but without dependency resolution)
        assert isinstance(deserialized, Parameter)
        assert deserialized.name == 'b'
        assert deserialized.independent is True  # Base path doesn't handle dependencies

    @pytest.mark.parametrize(
        'order',
        [['x', 'y', 'z'], ['z', 'x', 'y'], ['y', 'z', 'x'], ['z', 'y', 'x']],
        ids=[
            'normal_order',
            'dependent_first',
            'mixed_order',
            'dependent_first_reverse',
        ],
    )
    def test_serializer_id_system_order_independence(self, clear_global_map, order):
        """Test that dependency IDs allow parameters to be loaded in any order."""
        # WHEN
        # Create parameters with dependencies
        x = Parameter(name='x', value=5.0, unit='m', min=0, max=20)
        y = Parameter(name='y', value=10.0, unit='m', min=0, max=30)

        z = Parameter.from_dependency(
            name='z',
            dependency_expression='x * y',
            dependency_map={'x': x, 'y': y},
            unit='m^2',
        )

        # Verify original functionality
        assert z.value == 50.0  # 5 * 10

        # Get dependency IDs
        x_dep_id = x._DescriptorNumber__serializer_id
        y_dep_id = y._DescriptorNumber__serializer_id

        # Serialize all parameters
        params_data = {'x': x.as_dict(), 'y': y.as_dict(), 'z': z.as_dict()}

        # Verify dependency IDs are in serialized data
        assert params_data['x']['__serializer_id'] == x_dep_id
        assert params_data['y']['__serializer_id'] == y_dep_id
        assert '__serializer_id' not in params_data['z']
        assert '_dependency_map_serializer_ids' in params_data['z']

        # THEN
        global_object.map._clear()
        new_params = {}

        # Load in the specified order
        for name in order:
            new_params[name] = Parameter.from_dict(params_data[name])

        # EXPECT
        # Verify dependency IDs are preserved
        assert new_params['x']._DescriptorNumber__serializer_id == x_dep_id
        assert new_params['y']._DescriptorNumber__serializer_id == y_dep_id

        # Resolve dependencies
        resolve_all_parameter_dependencies(new_params)

        # Verify functionality regardless of loading order
        assert new_params['z'].independent is False
        assert new_params['z'].value == 50.0

        # Test dependency updates still work
        new_params['x'].value = 6.0
        assert new_params['z'].value == 60.0  # 6 * 10

        new_params['y'].value = 8.0
        assert new_params['z'].value == 48.0  # 6 * 8

    def test_deserialize_and_resolve_parameters_helper(self, clear_global_map):
        """Test the convenience helper function for deserialization and dependency resolution."""
        # Create test parameters with dependencies
        a = Parameter(name='a', value=2.0, unit='m', min=0, max=10)
        b = Parameter(name='b', value=3.0, unit='m', min=0, max=10)

        c = Parameter.from_dependency(
            name='c',
            dependency_expression='a + b',
            dependency_map={'a': a, 'b': b},
            unit='m',
        )

        # Verify original dependency works
        assert c.value == 5.0  # 2 + 3

        # Serialize all parameters
        params_data = {'a': a.as_dict(), 'b': b.as_dict(), 'c': c.as_dict()}

        # Clear global map
        global_object.map._clear()

        # Use the helper function instead of manual deserialization + resolution
        new_params = deserialize_and_resolve_parameters(params_data)

        # Verify all parameters are correctly deserialized and dependencies resolved
        assert len(new_params) == 3
        assert 'a' in new_params
        assert 'b' in new_params
        assert 'c' in new_params

        # Check that independent parameters work
        assert new_params['a'].name == 'a'
        assert new_params['a'].value == 2.0
        assert new_params['a'].independent is True

        assert new_params['b'].name == 'b'
        assert new_params['b'].value == 3.0
        assert new_params['b'].independent is True

        # Check that dependent parameter is properly resolved
        assert new_params['c'].name == 'c'
        assert new_params['c'].value == 5.0  # 2 + 3
        assert new_params['c'].independent is False

        # Verify dependency still works after helper function
        new_params['a'].value = 10.0
        assert new_params['c'].value == 13.0  # 10 + 3

        # Verify no pending dependencies remain
        pending = get_parameters_with_pending_dependencies(new_params)
        assert len(pending) == 0
