import pytest

from easyscience.Objects.variable.descriptor_bool import DescriptorBool
from easyscience import global_object

class TestDescriptorBool:
    @pytest.fixture
    def descriptor(self):
        descriptor = DescriptorBool(
            name="name",
            value=True,
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )
        return descriptor
    
    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_init(self, descriptor: DescriptorBool):
        # When Then Expect
        assert descriptor._bool_value == True

        # From super
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"

    @pytest.mark.parametrize("bool_value", ["string", 0, 1.0])
    def test_init_bool_value_type_exception(self, bool_value):

        # When Then Expect
        with pytest.raises(ValueError):
            DescriptorBool(
                name="name",
                value=bool_value,
                description="description",
                url="url",
                display_name="display_name",
                parent=None,
            )

    def test_value(self, descriptor: DescriptorBool):
        # When Then Expect
        assert descriptor.value == True

    def test_set_value(self, descriptor: DescriptorBool):
        # When Then
        descriptor.value = False

        # Expect
        assert descriptor._bool_value == False

    @pytest.mark.parametrize("bool_value", ["string", 0, 0.0])
    def test_set_value_type_exception(self, descriptor: DescriptorBool, bool_value):
        # When Then Expect
        with pytest.raises(TypeError):
            descriptor.value = bool_value

    def test_repr(self, descriptor: DescriptorBool):
        # When Then
        repr_str = str(descriptor)

        # Expect
        assert repr_str == "<DescriptorBool 'name': True>"

    def test_copy(self, descriptor: DescriptorBool):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorBool
        assert descriptor_copy._bool_value == descriptor._bool_value

    def test_as_data_dict(self, clear, descriptor: DescriptorBool):
        # When Then
        descriptor_dict = descriptor.as_data_dict()

        # Expect
        assert descriptor_dict == {
            "name": "name",
            "value": True,
            "description": "description",
            "url": "url",
            "display_name": "display_name",
            "unique_name": "DescriptorBool_0"
        }