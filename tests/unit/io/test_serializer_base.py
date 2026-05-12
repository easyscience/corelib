# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import datetime
from enum import Enum
from typing import Any
from typing import List
from typing import Optional
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

from easyscience import DescriptorNumber
from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import NewBase
from easyscience.io import SerializerBase
from easyscience.io import SerializerComponent


class TestEnum(Enum):
    TEST_VALUE = 'test'
    ANOTHER_VALUE = 42


class MockSerializerComponent(SerializerComponent):
    """Mock serializer component for testing"""

    def __init__(
        self, name: str = 'test', value: int = 1, optional_param: Optional[str] = None, **kwargs
    ):
        self.name = name
        self.value = value
        self.optional_param = optional_param
        self._kwargs = kwargs
        self.unique_name = f'mock_{name}'
        self._global_object = True


class MockSerializerWithConvertToDict(SerializerComponent):
    """Mock with custom _convert_to_dict method"""

    def __init__(self, name: str = 'test', value: int = 1):
        self.name = name
        self.value = value
        self.unique_name = f'custom_{name}'
        self._global_object = True

    def _convert_to_dict(self, d, encoder, skip=None, **kwargs):
        d['custom_field'] = 'added_by_convert_to_dict'
        return d


class ConcreteSerializer(SerializerBase):
    """Concrete implementation for testing abstract methods"""

    def encode(self, obj: SerializerComponent, skip: Optional[List[str]] = None, **kwargs) -> Any:
        return self._convert_to_dict(obj, skip=skip, **kwargs)

    @classmethod
    def decode(cls, obj: Any) -> Any:
        return cls._convert_from_dict(obj)


class TestSerializerBase:
    @pytest.fixture
    def clear(self):
        """Clear everything before and after each test"""
        global_object.map._clear()
        if global_object.stack:
            global_object.stack.clear()
        yield
        global_object.map._clear()
        if global_object.stack:
            global_object.stack.clear()

    @pytest.fixture
    def serializer(self):
        return ConcreteSerializer()

    @pytest.fixture
    def mock_obj(self):
        return MockSerializerComponent('test_obj', 42, 'optional_value')

    def test_abstract_methods_are_placeholders(self):
        """Test that SerializerBase can be instantiated but abstract methods just pass"""
        # SerializerBase can actually be instantiated (abstract methods just pass)
        base = SerializerBase()

        # Abstract methods don't raise exceptions - they just pass/return None
        result = base.encode(Mock())
        assert result is None

        result = SerializerBase.decode({})
        assert result is None

    def test_get_arg_spec(self):
        """Test get_arg_spec static method"""

        def example_func(self, arg1, arg2, arg3='default'):
            pass

        spec, args = SerializerBase.get_arg_spec(example_func)

        assert args == ['arg1', 'arg2', 'arg3']
        assert hasattr(spec, 'args')
        assert hasattr(spec, 'defaults')

    def test_encode_objs_datetime(self):
        """Test _encode_objs with datetime objects"""
        dt = datetime.datetime(2023, 10, 17, 14, 30, 45, 123456)
        result = SerializerBase._encode_objs(dt)

        expected = {
            '@module': 'datetime',
            '@class': 'datetime',
            'string': '2023-10-17 14:30:45.123456',
        }
        assert result == expected

    def test_encode_objs_numpy_array_real(self):
        """Test _encode_objs with real numpy arrays"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = SerializerBase._encode_objs(arr)

        expected = {'@module': 'numpy', '@class': 'array', 'dtype': 'int32', 'data': [1, 2, 3]}
        assert result == expected

    def test_encode_objs_numpy_array_complex(self):
        """Test _encode_objs with complex numpy arrays"""
        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        result = SerializerBase._encode_objs(arr)

        expected = {
            '@module': 'numpy',
            '@class': 'array',
            'dtype': 'complex128',
            'data': [[1.0, 3.0], [2.0, 4.0]],  # [real, imag]
        }
        assert result == expected

    def test_encode_objs_numpy_scalar(self):
        """Test _encode_objs with numpy scalars"""
        scalar = np.int32(42)
        result = SerializerBase._encode_objs(scalar)

        assert result == 42
        assert isinstance(result, int)

    def test_encode_objs_json_serializable(self):
        """Test _encode_objs with JSON serializable objects"""
        # Should return the object unchanged if it's JSON serializable
        data = {'key': 'value', 'number': 42}
        result = SerializerBase._encode_objs(data)
        assert result == data

    def test_encode_objs_non_serializable(self):
        """Test _encode_objs with non-serializable objects"""

        class NonSerializable:
            def __init__(self):
                self.value = 'test'

        obj = NonSerializable()
        result = SerializerBase._encode_objs(obj)
        # Should return the object unchanged if not serializable
        assert result is obj

    def test_convert_from_dict_datetime(self):
        """Test _convert_from_dict with datetime objects"""
        dt_dict = {
            '@module': 'datetime',
            '@class': 'datetime',
            'string': '2023-10-17 14:30:45.123456',
        }

        result = SerializerBase._convert_from_dict(dt_dict)
        expected = datetime.datetime(2023, 10, 17, 14, 30, 45, 123456)
        assert result == expected

    def test_convert_from_dict_datetime_no_microseconds(self):
        """Test _convert_from_dict with datetime without microseconds"""
        dt_dict = {'@module': 'datetime', '@class': 'datetime', 'string': '2023-10-17 14:30:45'}

        result = SerializerBase._convert_from_dict(dt_dict)
        expected = datetime.datetime(2023, 10, 17, 14, 30, 45)
        assert result == expected

    def test_convert_from_dict_numpy_array_real(self):
        """Test _convert_from_dict with real numpy arrays"""
        arr_dict = {'@module': 'numpy', '@class': 'array', 'dtype': 'int32', 'data': [1, 2, 3]}

        result = SerializerBase._convert_from_dict(arr_dict)
        expected = np.array([1, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_convert_from_dict_numpy_array_complex(self):
        """Test _convert_from_dict with complex numpy arrays"""
        arr_dict = {
            '@module': 'numpy',
            '@class': 'array',
            'dtype': 'complex128',
            'data': [[1.0, 3.0], [2.0, 4.0]],
        }

        result = SerializerBase._convert_from_dict(arr_dict)
        expected = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == expected.dtype

    def test_convert_from_dict_easyscience_object(self, clear):
        """Test _convert_from_dict with EasyScience objects"""
        param_dict = {
            '@module': 'easyscience.variable.parameter',
            '@class': 'Parameter',
            '@version': '0.6.0',
            'name': 'test_param',
            'value': 5.0,
            'unit': 'm',
            'variance': 0.1,
            'min': 0.0,
            'max': 10.0,
            'fixed': False,
            'url': '',
            'description': '',
            'display_name': 'test_param',
        }

        result = SerializerBase._convert_from_dict(param_dict)
        assert isinstance(result, Parameter)
        assert result.name == 'test_param'
        assert result.value == 5.0
        assert str(result.unit) == 'm'

    def test_convert_from_dict_list(self):
        """Test _convert_from_dict with lists"""
        data = [
            {'@module': 'datetime', '@class': 'datetime', 'string': '2023-10-17 14:30:45'},
            {'key': 'value'},
            42,
        ]

        result = SerializerBase._convert_from_dict(data)
        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], datetime.datetime)
        assert result[1] == {'key': 'value'}
        assert result[2] == 42

    def test_convert_from_dict_regular_dict(self):
        """Test _convert_from_dict with regular dictionaries"""
        data = {'key': 'value', 'number': 42}
        result = SerializerBase._convert_from_dict(data)
        assert result == data

    def test_convert_to_dict_basic(self, serializer, mock_obj, clear):
        """Test _convert_to_dict with basic object"""
        result = serializer._convert_to_dict(mock_obj)

        assert '@module' in result
        assert '@class' in result
        assert '@version' in result
        assert result['name'] == 'test_obj'
        assert result['value'] == 42
        assert result['optional_param'] == 'optional_value'
        assert result['unique_name'] == 'mock_test_obj'

    def test_convert_to_dict_with_skip(self, serializer, mock_obj, clear):
        """Test _convert_to_dict with skip parameter"""
        result = serializer._convert_to_dict(mock_obj, skip=['value', 'optional_param'])

        assert 'name' in result
        assert 'value' not in result
        assert 'optional_param' not in result

    def test_convert_to_dict_with_custom_convert_to_dict(self, serializer, clear):
        """Test _convert_to_dict with custom _convert_to_dict method"""
        obj = MockSerializerWithConvertToDict('custom_test', 5)
        result = serializer._convert_to_dict(obj)

        assert result['custom_field'] == 'added_by_convert_to_dict'
        assert result['name'] == 'custom_test'
        assert result['value'] == 5

    def test_descriptor_encode_skips_parent(self, clear):
        """Test DescriptorNumber encode skips parent references."""
        descriptor = DescriptorNumber('child', 1.0)  # type: ignore

        result = descriptor.encode()

        assert 'parent' not in result

    def test_parameter_encode_skips_callback(self, clear):
        """Test Parameter encode skips callback properties."""
        parameter = Parameter('test_parameter', 1.0, callback=property())

        result = parameter.encode()

        assert 'callback' not in result

    def test_convert_to_dict_with_enum_object(self, serializer, clear):
        """Test _convert_to_dict when the object itself is an enum"""

        # Test that enum values in objects remain as enums without full_encode
        class MockObjWithEnum(SerializerComponent):
            def __init__(self, name: str, enum_val: TestEnum):
                self.name = name
                self.enum_val = enum_val
                self.unique_name = f'obj_{name}'
                self._global_object = True

        obj = MockObjWithEnum('test', TestEnum.TEST_VALUE)
        result = serializer._convert_to_dict(obj)

        # The enum field should remain as an enum object (not encoded as dict)
        assert isinstance(result['enum_val'], TestEnum)
        assert result['enum_val'] == TestEnum.TEST_VALUE

    def test_convert_to_dict_full_encode(self, serializer, clear):
        """Test _convert_to_dict with full_encode=True"""
        dt = datetime.datetime(2023, 10, 17, 14, 30, 45)

        class MockObjWithDateTime(SerializerComponent):
            def __init__(self, name: str, dt: datetime.datetime):
                self.name = name
                self.dt = dt
                self.unique_name = f'dt_{name}'
                self._global_object = True

        obj = MockObjWithDateTime('test', dt)
        result = serializer._convert_to_dict(obj, full_encode=True)

        assert isinstance(result['dt'], dict)
        assert result['dt']['@module'] == 'datetime'
        assert result['dt']['@class'] == 'datetime'

    def test_convert_to_dict_without_global_object(self, serializer):
        """Test _convert_to_dict with object without _global_object"""

        class MockObjNoGlobal(SerializerComponent):
            def __init__(self, name: str):
                self.name = name

        obj = MockObjNoGlobal('test')
        result = serializer._convert_to_dict(obj)

        assert result['name'] == 'test'
        assert 'unique_name' not in result

    def test_convert_to_dict_with_arg_spec(self, serializer, clear):
        """Test _convert_to_dict with custom _arg_spec"""

        class MockObjCustomArgSpec(SerializerComponent):
            def __init__(self, name: str, value: int, extra: str = 'default'):
                self.name = name
                self.value = value
                self.extra = extra
                self._arg_spec = ['name', 'value']  # Skip 'extra'
                self.unique_name = f'custom_spec_{name}'
                self._global_object = True

        obj = MockObjCustomArgSpec('test', 42, 'not_default')
        result = serializer._convert_to_dict(obj)

        assert 'name' in result
        assert 'value' in result
        assert 'extra' not in result

    def test_recursive_encoder_with_lists(self, serializer, clear):
        """Test _recursive_encoder with lists"""
        mock_obj = MockSerializerComponent('list_test', 1)
        data = [mock_obj, 'string', 42, {'key': 'value'}]

        result = serializer._recursive_encoder(data)

        assert isinstance(result, list)
        assert len(result) == 4
        assert isinstance(result[0], dict)  # Encoded mock object
        assert result[0]['name'] == 'list_test'
        assert result[1] == 'string'
        assert result[2] == 42
        assert result[3] == {'key': 'value'}

    def test_recursive_encoder_with_dicts(self, serializer, clear):
        """Test _recursive_encoder with dictionaries"""
        mock_obj = MockSerializerComponent('dict_test', 2)
        data = {'obj': mock_obj, 'simple': 'value'}

        result = serializer._recursive_encoder(data)

        assert isinstance(result, dict)
        assert isinstance(result['obj'], dict)  # Encoded mock object
        assert result['obj']['name'] == 'dict_test'
        assert result['simple'] == 'value'

    def test_recursive_encoder_with_tuples(self, serializer, clear):
        """Test _recursive_encoder with tuples"""
        mock_obj = MockSerializerComponent('tuple_test', 3)
        data = (mock_obj, 'string', 42)

        result = serializer._recursive_encoder(data)

        assert isinstance(result, list)  # Tuples become lists
        assert len(result) == 3
        assert isinstance(result[0], dict)  # Encoded mock object
        assert result[0]['name'] == 'tuple_test'

    def test_recursive_encoder_builtin_encode_method(self, serializer):
        """Test _recursive_encoder doesn't encode builtin objects with encode method"""
        # Strings have an encode method but are builtin
        data = ['test_string', b'bytes']

        result = serializer._recursive_encoder(data)

        assert result == ['test_string', b'bytes']

    def test_recursive_encoder_with_mutable_sequence(self, serializer, clear):
        """Test _recursive_encoder with MutableSequence objects"""
        from easyscience.base_classes import CollectionBase

        d0 = DescriptorNumber('a', 0)  # type: ignore
        d1 = DescriptorNumber('b', 1)  # type: ignore
        collection = CollectionBase('test_collection', d0, d1)

        result = serializer._recursive_encoder(collection)

        assert isinstance(result, dict)
        assert result['@class'] == 'CollectionBase'
        assert 'data' in result

    @patch('easyscience.io.serializer_base.import_module')
    def test_convert_to_dict_no_version(self, mock_import, serializer, clear):
        """Test _convert_to_dict when module has no __version__"""
        mock_module = Mock()
        del mock_module.__version__  # Remove version attribute
        mock_import.return_value = mock_module

        mock_obj = MockSerializerComponent('no_version', 1)
        result = serializer._convert_to_dict(mock_obj)

        assert result['@version'] is None

    @patch('easyscience.io.serializer_base.import_module')
    def test_convert_to_dict_import_error(self, mock_import, serializer, clear):
        """Test _convert_to_dict when import_module raises ImportError"""
        mock_import.side_effect = ImportError('Module not found')

        mock_obj = MockSerializerComponent('import_error', 1)
        result = serializer._convert_to_dict(mock_obj)

        assert result['@version'] is None

    def test_convert_to_dict_attribute_error_handling(self, serializer, clear):
        """Test _convert_to_dict handles AttributeError for missing attributes"""

        class MockObjMissingAttrs(SerializerComponent):
            def __init__(self, name: str, missing_param: str = 'default'):
                self.name = name
                self.unique_name = f'missing_{name}'
                self._global_object = True
                # Don't set missing_param attribute to trigger AttributeError

        obj = MockObjMissingAttrs('test')

        with pytest.raises(NotImplementedError, match='Unable to automatically determine as_dict'):
            serializer._convert_to_dict(obj)

    def test_convert_to_dict_with_kwargs_attribute(self, serializer, clear):
        """Test _convert_to_dict with _kwargs attribute handling"""

        class MockObjWithKwargs(SerializerComponent):
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value
                self.unique_name = f'kwargs_{name}'
                self._global_object = True
                # Set up _kwargs to test the kwargs handling path
                self._kwargs = {'extra_param': 'extra_value'}

        obj = MockObjWithKwargs('test', 42)
        result = serializer._convert_to_dict(obj)

        # The extra_param from _kwargs should be included
        assert result['extra_param'] == 'extra_value'

    def test_convert_to_dict_varargs_handling(self, serializer, clear):
        """Test _convert_to_dict with varargs (*args) handling"""

        class MockObjWithVarargs(SerializerComponent):
            def __init__(self, name: str, *args):
                self.name = name
                self.args = args
                self.unique_name = f'varargs_{name}'
                self._global_object = True

        obj = MockObjWithVarargs('test', 'arg1', 'arg2', 'arg3')
        result = serializer._convert_to_dict(obj)

        assert result['name'] == 'test'
        if 'args' in result:
            assert result['args'] == ('arg1', 'arg2', 'arg3')

    def test_encode_objs_edge_cases(self):
        """Test _encode_objs with edge cases"""
        # Test with None
        assert SerializerBase._encode_objs(None) is None

        # Test with empty numpy array
        empty_arr = np.array([])
        result = SerializerBase._encode_objs(empty_arr)
        assert result['@module'] == 'numpy'
        assert result['data'] == []

    def test_convert_from_dict_edge_cases(self):
        """Test _convert_from_dict with edge cases"""
        # Test with None
        assert SerializerBase._convert_from_dict(None) is None

        # Test with empty dict
        assert SerializerBase._convert_from_dict({}) == {}

        # Test with empty list
        assert SerializerBase._convert_from_dict([]) == []

        # Test with bson.objectid (should not be processed)
        bson_dict = {'@module': 'bson.objectid', '@class': 'ObjectId', 'value': 'some_id'}
        result = SerializerBase._convert_from_dict(bson_dict)
        assert result == bson_dict

    def test_concrete_serializer_implementation(self, clear):
        """Test that ConcreteSerializer works correctly"""
        serializer = ConcreteSerializer()
        mock_obj = MockSerializerComponent('concrete_test', 100)

        # Test encode
        encoded = serializer.encode(mock_obj)
        assert isinstance(encoded, dict)
        assert encoded['name'] == 'concrete_test'
        assert encoded['value'] == 100

        # Test decode
        global_object.map._clear()  # Clear before decode
        decoded = ConcreteSerializer.decode(encoded)
        assert isinstance(decoded, MockSerializerComponent)
        assert decoded.name == 'concrete_test'
        assert decoded.value == 100

    def test_get_arg_spec_with_complex_function(self):
        """Test get_arg_spec with more complex function signatures"""

        def complex_func(self, required_arg, optional_arg='default', *args, **kwargs):
            pass

        spec, args = SerializerBase.get_arg_spec(complex_func)

        assert args == ['required_arg', 'optional_arg']
        assert spec.varargs == 'args'
        assert spec.varkw == 'kwargs'
        assert spec.defaults == ('default',)

    @patch('easyscience.io.serializer_base.np', None)
    def test_encode_objs_without_numpy(self):
        """Test _encode_objs when numpy is not available"""
        # This test patches np to None to simulate numpy not being installed
        dt = datetime.datetime(2023, 10, 17, 14, 30, 45)
        result = SerializerBase._encode_objs(dt)

        expected = {'@module': 'datetime', '@class': 'datetime', 'string': '2023-10-17 14:30:45'}
        assert result == expected

    def test_recursive_encoder_with_nested_structures(self, serializer, clear):
        """Test _recursive_encoder with deeply nested structures"""
        mock_obj = MockSerializerComponent('nested', 1)

        data = {'level1': {'level2': [mock_obj, {'level3': mock_obj}]}}

        result = serializer._recursive_encoder(data)

        assert isinstance(result, dict)
        assert isinstance(result['level1'], dict)
        assert isinstance(result['level1']['level2'], list)
        assert isinstance(result['level1']['level2'][0], dict)  # Encoded object
        assert result['level1']['level2'][0]['name'] == 'nested'
        assert isinstance(result['level1']['level2'][1]['level3'], dict)  # Encoded object

    def test_import_class(self):
        # When Then
        cls = SerializerBase._import_class(module_name='easyscience', class_name='Parameter')
        # Expect
        assert cls is Parameter

    def test_import_class_missing_module(self):
        # When Then Expect
        with pytest.raises(ImportError):
            SerializerBase._import_class(module_name='non_existent_module', class_name='Parameter')

    def test_import_class_missing_class(self):
        # When Then Expect
        with pytest.raises(ValueError):
            SerializerBase._import_class(module_name='easyscience', class_name='NonExistentClass')

    @pytest.mark.parametrize(
        'dict, expected',
        [
            (
                {'@module': 'easyscience', '@class': 'Parameter', 'name': 'param1', 'value': 10.0},
                True,
            ),
            (
                {
                    '@module': 'numpy',
                    '@class': 'array',
                    'unique_name': 'unique1',
                    'display_name': 'Display 1',
                },
                False,
            ),
        ],
        ids=['valid_easyscience_dict', 'invalid_numpy_dict'],
    )
    def test_is_serialized_easyscience_object_false(self, dict, expected):
        # When Then
        result = SerializerBase._is_serialized_easyscience_object(dict)
        # Expect
        assert result is expected

    def test_deserialize_value_non_easyscience(self):
        # When
        serialized_dict = {'@module': 'numpy', '@class': 'array', 'dtype': 'int64', 'data': [0, 1]}
        # Then
        obj = SerializerBase._deserialize_value(serialized_dict)
        # Expect
        assert isinstance(obj, np.ndarray)
        assert obj.dtype == np.int64

    def test_deserialize_value_easyscience_uses_from_dict(self, monkeypatch):
        # When
        serialized_dict = {
            '@module': 'easyscience.base_classes',
            '@class': 'NewBase',
            'display_name': 'test',
        }
        monkeypatch.setattr(NewBase, 'from_dict', MagicMock())
        # Then
        obj = SerializerBase._deserialize_value(serialized_dict)
        # Expect
        NewBase.from_dict.assert_called_once_with(serialized_dict)

    def test_deserialize_value_easyscience_no_from_dict(self, monkeypatch):
        # When
        serialized_dict = {
            '@module': 'easyscience.base_classes',
            '@class': 'NewBase',
            'display_name': 'test',
        }
        monkeypatch.delattr(NewBase, 'from_dict')
        monkeypatch.setattr(SerializerBase, '_convert_from_dict', MagicMock())
        # Then
        obj = SerializerBase._deserialize_value(serialized_dict)
        # Expect
        SerializerBase._convert_from_dict.assert_called_once_with(serialized_dict)

    def test_deserialize_value_easyscience_import_error(self, monkeypatch):
        # When
        serialized_dict = {
            '@module': 'easyscience.base_classes',
            '@class': 'NewBase',
            'display_name': 'test',
        }
        monkeypatch.setattr(SerializerBase, '_import_class', MagicMock(side_effect=ImportError()))
        monkeypatch.setattr(SerializerBase, '_convert_from_dict', MagicMock())
        # Then
        obj = SerializerBase._deserialize_value(serialized_dict)
        # Expect
        SerializerBase._convert_from_dict.assert_called_once_with(serialized_dict)

    def test_deserialize_dict(self, monkeypatch):
        # When
        serialized_dict = {
            '@module': 'easyscience',
            '@class': 'ParameterContainer',
            'param1': {
                '@module': 'easyscience',
                '@class': 'Parameter',
                'name': 'param1',
                'value': 10.0,
            },
            'array1': {'@module': 'numpy', '@class': 'array', 'dtype': 'int64', 'data': [0, 1]},
        }
        monkeypatch.setattr(
            SerializerBase,
            '_deserialize_value',
            MagicMock(
                side_effect=[
                    Parameter(name='param1', value=10.0),
                    np.array([0, 1], dtype=np.int64),
                ]
            ),
        )
        # Then
        result = SerializerBase.deserialize_dict(serialized_dict)
        # Expect
        SerializerBase._deserialize_value.assert_any_call(serialized_dict['param1'])
        SerializerBase._deserialize_value.assert_any_call(serialized_dict['array1'])
        assert isinstance(result['param1'], Parameter)
        assert isinstance(result['array1'], np.ndarray)
        assert result['array1'].dtype == np.int64
