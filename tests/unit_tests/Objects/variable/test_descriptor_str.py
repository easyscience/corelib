import pytest

from easyscience.Objects.variable.descriptor_str import DescriptorStr
from easyscience import global_object

class TestDescriptorStr:
    @pytest.fixture
    def descriptor(self):
        descriptor = DescriptorStr(
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

    def test_init(self, descriptor: DescriptorStr):
        # When Then Expect
        assert descriptor._string == "string"

        # From super
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"

    @pytest.mark.parametrize("string", [True, 0, 1.0])
    def test_init_string_type_exception(self, string):
        # When Then Expect
        with pytest.raises(ValueError):
            DescriptorStr(
                name="name",
                value=string,
                description="description",
                url="url",
                display_name="display_name",
                parent=None,
            )

    def test_value(self, descriptor: DescriptorStr):
        # When Then Expect
        assert descriptor.value == "string"

    def test_set_value(self, descriptor: DescriptorStr):
        # When Then
        descriptor.value = "new_string"

        # Expect
        assert descriptor._string == "new_string"

    @pytest.mark.parametrize("string", [True, 0, 1.0])
    def test_set_value_type_exception(self, descriptor: DescriptorStr, string):
        # When Then Expect
        with pytest.raises(ValueError):
            descriptor.value = string

    def test_repr(self, descriptor: DescriptorStr):
        # When Then
        repr_str = str(descriptor)

        # Expect
        assert repr_str == "<DescriptorStr 'name': string>"

    def test_copy(self, descriptor: DescriptorStr):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorStr
        assert descriptor_copy._string == descriptor._string

    def test_as_data_dict(self, clear, descriptor: DescriptorStr):
        # When Then
        descriptor_dict = descriptor.as_data_dict()

        # Expect
        assert descriptor_dict == {
            "name": "name",
            "value": "string",
            "description": "description",
            "url": "url",
            "display_name": "display_name",
            "unique_name": "DescriptorStr_0"
        }