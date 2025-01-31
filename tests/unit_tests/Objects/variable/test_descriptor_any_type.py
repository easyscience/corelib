import pytest
import numpy as np

from easyscience.Objects.variable.descriptor_any_type import DescriptorAnyType
from easyscience import global_object

class TestDescriptorAnyType:
    @pytest.fixture
    def descriptor(self):
        descriptor = DescriptorAnyType(
            name="name",
            value="string",
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )
        return descriptor
    
    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_init(self, descriptor: DescriptorAnyType):
        # When Then Expect
        assert descriptor._value == "string"

        # From super
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"

    def test_value(self, descriptor: DescriptorAnyType):
        # When Then Expect
        assert descriptor.value == "string"

    
    @pytest.mark.parametrize("value", [True, "new_string", 1.0, np.array([1, 2, 3]),{"key": "value"}])
    def test_set_value(self, descriptor: DescriptorAnyType,value):
        # When Then
        descriptor.value = value

        # Expect
        if isinstance(value, np.ndarray):
            assert np.array_equal(descriptor._value, value)
        else:
            assert descriptor._value == value

    @pytest.mark.parametrize(
        "value, expected",
        [
            (True, "True"),
            ("new_string", "'new_string'"),
            (1.0, "1.0"),
            (np.array([1, 2, 3]), "array([1, 2, 3])"),
            ({"key": "value"}, "{'key': 'value'}")
        ]
    )
    def test_repr(self, descriptor: DescriptorAnyType, value, expected):
        # Set the descriptor value
        descriptor.value = value

        # When Then
        repr_str = str(descriptor)

        print(repr_str)

        # Expect
        assert repr_str == f"<DescriptorAnyType 'name': {expected}>"

    def test_copy(self, descriptor: DescriptorAnyType):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorAnyType
        assert descriptor_copy._value == descriptor._value

    def test_as_data_dict(self, clear, descriptor: DescriptorAnyType):
        # When Then
        descriptor_dict = descriptor.as_data_dict()

        # Expect
        assert descriptor_dict == {
            "name": "name",
            "value": "string",
            "description": "description",
            "url": "url",
            "display_name": "display_name",
            "unique_name": "DescriptorAnyType_0"
        }