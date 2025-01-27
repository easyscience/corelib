import pytest
import numpy as np
import scipp as sc
from scipp import UnitError
from easyscience.Objects.variable.descriptor_array import DescriptorArray

class TestDescriptorArray:
    @pytest.fixture
    def descriptor(self):
        values = np.array([[1., 2.], [3., 4.]])
        variances = np.array([[0.1, 0.2], [0.3, 0.4]])
        return DescriptorArray(
            name="name",
            value=values,
            unit="m",
            variance=variances,
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )

    def test_init(self, descriptor: DescriptorArray):
        # When Then Expect
        assert np.array_equal(descriptor._array.values, np.array([[1., 2.], [3., 4.]]))
        assert np.array_equal(descriptor._array.variances, np.array([[0.1, 0.2], [0.3, 0.4]]))
        assert descriptor._array.unit == "m"

        # From super
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"

    def test_init_invalid_value_type(self):
        with pytest.raises(TypeError):
            DescriptorArray(name="name", value=1, unit="m")

    def test_init_invalid_variance_type(self):
        with pytest.raises(TypeError):
            DescriptorArray(name="name", value=[[1, 2]], unit="m", variance=1)

    def test_init_variance_shape_mismatch(self):
        with pytest.raises(ValueError):
            DescriptorArray(
                name="name",
                value=[[1, 2]],
                unit="m",
                variance=[[0.1]],
            )

    def test_init_invalid_unit(self):
        with pytest.raises(UnitError):
            DescriptorArray(name="name", value=[[1, 2]], unit="unknown")

    def test_from_scipp(self):
        # When
        scipp_array = sc.array(dims=["row", "column"], values=[[1., 2.]], unit="m", variances=[[0.1, 0.2]])
        descriptor = DescriptorArray.from_scipp(name="name", full_value=scipp_array)

        # Then
        assert np.array_equal(descriptor._array.values, [[1., 2.]])
        assert np.array_equal(descriptor._array.variances, [[0.1, 0.2]])
        assert descriptor._array.unit == "m"

    def test_full_value(self, descriptor: DescriptorArray):
        # When Then
        assert descriptor.full_value.unit == "m"
        assert np.array_equal(descriptor.full_value.values, [[1., 2.], [3., 4.]])
        assert np.array_equal(descriptor.full_value.variances, [[0.1, 0.2], [0.3, 0.4]])

    def test_set_full_value(self, descriptor: DescriptorArray):
        # When Then
        with pytest.raises(AttributeError):
            descriptor.full_value = sc.scalar(2, unit="m")

    def test_value_property(self, descriptor: DescriptorArray):
        # When Then
        assert np.array_equal(descriptor.value, [[1., 2.], [3., 4.]])

    def test_set_value(self, descriptor: DescriptorArray):
        # When
        new_value = np.array([[5, 6], [7, 8]])
        descriptor.value = new_value

        # Then
        assert np.array_equal(descriptor.value, new_value)

    def test_unit(self, descriptor: DescriptorArray):
        # When Then
        assert descriptor.unit == "m"

    def test_set_unit(self, descriptor: DescriptorArray):
        # When Then
        with pytest.raises(AttributeError):
            descriptor.unit = "cm"

    def test_variance(self, descriptor: DescriptorArray):
        # When Then
        assert np.array_equal(descriptor.variance, [[0.1, 0.2], [0.3, 0.4]])

    def test_set_variance(self, descriptor: DescriptorArray):
        # When
        new_variance = np.array([[0.5, 0.6], [0.7, 0.8]])
        descriptor.variance = new_variance

        # Then
        assert np.array_equal(descriptor.variance, new_variance)

    def test_error(self, descriptor: DescriptorArray):
        # When Then
        assert np.array_equal(descriptor.error, np.sqrt([[0.1, 0.2], [0.3, 0.4]]))

    def test_set_error(self, descriptor: DescriptorArray):
        # When
        new_error = np.array([[0.1, 0.2], [0.3, 0.4]])
        descriptor.error = new_error

        # Then
        assert np.array_equal(descriptor.variance, new_error**2)

    def test_convert_unit(self, descriptor: DescriptorArray):
        # When
        descriptor.convert_unit("cm")

        # Then
        assert descriptor.unit == "cm"
        assert np.array_equal(descriptor.value, [[100, 200], [300, 400]])

    def test_addition(self, descriptor: DescriptorArray):
        # When
        other = DescriptorArray("other", value=[[1, 1], [1, 1]], unit="m")
        result = descriptor + other

        # Then
        assert result.unit == "m"
        assert np.array_equal(result.value, [[2, 3], [4, 5]])

    def test_addition_with_scalar(self, descriptor: DescriptorArray):
        # When
        descriptor = DescriptorArray("descriptor", value=[[1., 2.], [3., 4.]])
        result = descriptor + 1.0

        # Then
        assert result.unit == "dimensionless"
        assert np.array_equal(result.value, [[2, 3], [4, 5]])

    def test_subtraction(self, descriptor: DescriptorArray):
        # When
        other = DescriptorArray("other", value=[[1, 1], [1, 1]], unit="m")
        result = descriptor - other

        # Then
        assert result.unit == "m"
        assert np.array_equal(result.value, [[0, 1], [2, 3]])

    def test_subtraction_with_scalar(self, descriptor: DescriptorArray):
        # When
        result = descriptor - 1.0

        # Then
        assert result.unit == "dimensionless"
        assert np.array_equal(result.value, [[0, 1], [2, 3]])

    def test_repr(self, descriptor: DescriptorArray):
        # When
        repr_str = repr(descriptor)

        # Then
        assert "DescriptorArray" in repr_str
        assert "values=[[1 2]" in repr_str
        assert "unit=m" in repr_str
