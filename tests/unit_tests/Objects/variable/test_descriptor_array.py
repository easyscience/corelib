import pytest
from unittest.mock import MagicMock
import scipp as sc
from scipp import UnitError

import numpy as np

from easyscience.Objects.variable.descriptor_array import DescriptorArray
from easyscience import global_object

class TestDescriptorArray:
    @pytest.fixture
    def descriptor(self):
        descriptor = DescriptorArray(
            name="name",
            value=[[1., 2.], [3., 4.]],
            unit="m",
            variance=[[0.1, 0.2], [0.3, 0.4]],
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )
        return descriptor

    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_init(self, descriptor: DescriptorArray):
        # When Then Expect
        assert np.array_equal(descriptor._array.values,np.array([[1., 2.], [3., 4.]]))
        assert descriptor._array.unit == "m"
        assert np.array_equal(descriptor._array.variances, np.array([[0.1, 0.2], [0.3, 0.4]]))

        # From super
        assert descriptor._name == "name"
        assert descriptor._description == "description"
        assert descriptor._url == "url"
        assert descriptor._display_name == "display_name"

    def test_init_sc_unit(self):
        # When Then
        descriptor = DescriptorArray(
            name="name",
            value=[[1., 2.], [3., 4.]],
            unit=sc.units.Unit("m"),
            variance=[[0.1, 0.2], [0.3, 0.4]],
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )

        # Expect
        assert np.array_equal(descriptor._array.values,np.array([[1., 2.], [3., 4.]]))
        assert descriptor._array.unit == "m"
        assert np.array_equal(descriptor._array.variances, np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_init_sc_unit_unknown(self):
        # When Then Expect
        with pytest.raises(UnitError):
            DescriptorArray(
                name="name",
                value=[[1., 2.], [3., 4.]],
                unit="unknown",
                variance=[[0.1, 0.2], [0.3, 0.4]],
                description="description",
                url="url",
                display_name="display_name",
                parent=None,
            )

    @pytest.mark.parametrize("value", [True, "string"])
    def test_init_value_type_exception(self, value):
        # When 

        # Then Expect
        with pytest.raises(TypeError):
            DescriptorArray(
                name="name",
                value=value,
                unit="m",
                variance=[[0.1, 0.2], [0.3, 0.4]],
                description="description",
                url="url",
                display_name="display_name",
                parent=None,
            )

    def test_init_variance_exception(self):
        # When 
        variance=[[-0.1, -0.2], [-0.3, -0.4]]
        # Then Expect
        with pytest.raises(ValueError):
            DescriptorArray(
                name="name",
                value=[[1., 2.], [3., 4.]],
                unit="m",
                variance=variance,
                description="description",
                url="url",
                display_name="display_name",
                parent=None,
            )

    # test from_scipp
    def test_from_scipp(self):
        # When
        full_value = sc.array(dims=['row','column'],values=[[1,2],[3,4]], unit='m')
        # Then
        descriptor = DescriptorArray.from_scipp(name="name", full_value=full_value)

        # Expect
        assert np.array_equal(descriptor._array.values,[[1,2],[3,4]])
        assert descriptor._array.unit == "m"
        assert descriptor._array.variances == None

    # @pytest.mark.parametrize("full_value", [sc.array(values=[1,2], dims=["x"]), sc.array(values=[[1], [2]], dims=["x","y"]), object(), 1, "string"], ids=["1D", "2D", "object", "int", "string"])
    # def test_from_scipp_type_exception(self, full_value):
    #     # When Then Expect
    #     with pytest.raises(TypeError):
    #         DescriptorArray.from_scipp(name="name", full_value=full_value)

    def test_full_value(self, descriptor: DescriptorArray):
        # When Then Expect
        assert descriptor.full_value == sc.array(dims=['row','column'],values=[[1,2],[3,4]], unit='m')
        
    def test_set_full_value(self, descriptor: DescriptorArray):
        with pytest.raises(AttributeError):
            descriptor.full_value = sc.array(dims=['row','column'],values=[[1,2],[3,4]], unit='s')

    def test_unit(self, descriptor: DescriptorArray):
        # When Then Expect
        assert descriptor.unit == 'm'
        
    def test_set_unit(self, descriptor: DescriptorArray):
        with pytest.raises(AttributeError):
            descriptor.unit = 's'

    def test_convert_unit(self, descriptor: DescriptorArray):
        # When  Then
        descriptor.convert_unit('mm')

        # Expect
        assert descriptor._array.unit == 'mm'
        print(descriptor._array.variances)
        assert np.array_equal(descriptor._array.values,[[1000,2000],[3000,4000]])
        assert np.array_equal(descriptor._array.variances,[[100000,200000],[300000,400000]])

    def test_variance(self, descriptor: DescriptorArray):
        # When Then Expect
        assert np.array_equal(descriptor._array.variances, np.array([[0.1, 0.2], [0.3, 0.4]]))

        
    def test_set_variance(self, descriptor: DescriptorArray):
        # When Then
        descriptor.variance = [[0.2, 0.3], [0.4, 0.5]]

        # Expect
        assert np.array_equal(descriptor.variance, np.array([[0.2, 0.3], [0.4, 0.5]]))

        assert np.array_equal(descriptor.error, np.sqrt(np.array([[0.2, 0.3], [0.4, 0.5]])))

    def test_error(self, descriptor: DescriptorArray):
        # When Then Expect
        assert np.array_equal(descriptor.error, np.sqrt(np.array([[0.1, 0.2], [0.3, 0.4]])))

        
    def test_set_error(self, descriptor: DescriptorArray):
        # When Then
        descriptor.error = np.sqrt(np.array([[0.2, 0.3], [0.4, 0.5]]))
        print(descriptor.variance)
        print(descriptor.error)
        # Expect
        assert np.allclose(descriptor.error, np.sqrt(np.array([[0.2, 0.3], [0.4, 0.5]])))
        assert np.allclose(descriptor.variance, np.array([[0.2, 0.3], [0.4, 0.5]]))


    def test_value(self, descriptor: DescriptorArray):
        # When Then Expect
        print(descriptor.value)
        
        assert np.array_equal(descriptor.value, np.array([[1, 2], [3, 4]]))

    def test_set_value(self, descriptor: DescriptorArray):
        # When Then
        descriptor.value = ([[0.2, 0.3], [0.4, 0.5]])
        # Expect
        assert np.array_equal(descriptor._array.values, np.array([[0.2, 0.3], [0.4, 0.5]]))

    def test_repr(self, descriptor: DescriptorArray):
        # When Then
        repr_str = str(descriptor)

        # Expect
        assert repr_str ==  "<DescriptorArray 'name': values=[[1. 2.], [3. 4.]], errors=[[0.3162 0.4472], [0.5477 0.6325]], unit=m>"

    def test_copy(self, descriptor: DescriptorArray):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorArray
        assert np.array_equal(descriptor_copy._array.values, descriptor._array.values)
        assert descriptor_copy._array.unit == descriptor._array.unit

    def test_as_data_dict(self, clear, descriptor: DescriptorArray):
        # When
        descriptor_dict = descriptor.as_data_dict()

        # Expected dictionary
        expected_dict = {
            "name": "name",
            "value": np.array([[1.0, 2.0], [3.0, 4.0]]),  # Use numpy array for comparison
            "unit": "m",
            "variance": np.array([[0.1, 0.2], [0.3, 0.4]]),  # Use numpy array for comparison
            "description": "description",
            "url": "url",
            "display_name": "display_name",
            "unique_name": "DescriptorArray_0",
        }

        # Then: Compare dictionaries key by key
        for key, expected_value in expected_dict.items():
            if isinstance(expected_value, np.ndarray):
                # Compare numpy arrays
                assert np.array_equal(descriptor_dict[key], expected_value), f"Mismatch for key: {key}"
            else:
                # Compare other values directly
                assert descriptor_dict[key] == expected_value, f"Mismatch for key: {key}"


 
    @pytest.mark.parametrize("unit_string, expected", [
        ("1e+9", "dimensionless"),
        ("1000", "dimensionless"),
        ("10dm^2", "m^2")],
        ids=["scientific_notation", "numbers", "unit_prefix"])
    def test_base_unit(self, unit_string, expected):
        # When
        descriptor = DescriptorArray(name="name", value=[[1.0, 2.0], [3.0, 4.0]], unit=unit_string)

        # Then
        base_unit = descriptor._base_unit()

        # Expect
        assert base_unit == expected

    @pytest.mark.parametrize("test, expected", [
        (DescriptorArray("test", 2, "m", 0.01,),   DescriptorArray("test + name", 3, "m", 0.11)),
        (DescriptorArray("test", 2, "cm", 0.01),   DescriptorArray("test + name", 102, "cm", 1000.01))],
        ids=["regular", "unit_conversion"])
    def test_addition(self, descriptor: DescriptorArray, test, expected):
        # When Then
        result = test + descriptor
# [[2.0, 3.0], [3.0, 4.0]]
        # Expect
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.array_equal(result.variance, expected.variance)
        
        assert descriptor.unit == 'm'

            # value=[[1., 2.], [3., 4.]],
            # unit="m",
            # variance=[[0.1, 0.2], [0.3, 0.4]],


    # def test_addition_with_array(self):
    #     # When 
    #     descriptor = DescriptorArray(name="name", value=1, variance=0.1)

    #     # Then
    #     result = descriptor + 1.0
    #     result_reverse = 1.0 + descriptor

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == 2.0
    #     assert result.unit == "dimensionless"
    #     assert result.variance == 0.1

    #     assert type(result_reverse) == DescriptorArray
    #     assert result_reverse.name == result_reverse.unique_name
    #     assert result_reverse.value == 2.0
    #     assert result_reverse.unit == "dimensionless"
    #     assert result_reverse.variance == 0.1

    # @pytest.mark.parametrize("test", [1.0, DescriptorArray("test", 2, "s",)], ids=["add_array_to_unit", "incompatible_units"])
    # def test_addition_exception(self, descriptor: DescriptorArray, test):
    #     # When Then Expect
    #     with pytest.raises(UnitError):
    #         result = descriptor + test
    #     with pytest.raises(UnitError):
    #         result_reverse = test + descriptor
        
    # @pytest.mark.parametrize("test, expected", [
    #     (DescriptorArray("test", 2, "m", 0.01,),   DescriptorArray("test - name", 1, "m", 0.11)),
    #     (DescriptorArray("test", 2, "cm", 0.01),   DescriptorArray("test - name", -98, "cm", 1000.01))],
    #     ids=["regular", "unit_conversion"])
    # def test_subtraction(self, descriptor: DescriptorArray, test, expected):
    #     # When Then
    #     result = test - descriptor

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == expected.value
    #     assert result.unit == expected.unit
    #     assert result.variance == expected.variance

    #     assert descriptor.unit == 'm'

    # def test_subtraction_with_array(self):
    #     # When 
    #     descriptor = DescriptorArray(name="name", value=2, variance=0.1)

    #     # Then
    #     result = descriptor - 1.0
    #     result_reverse = 1.0 - descriptor

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == 1.0
    #     assert result.unit == "dimensionless"
    #     assert result.variance == 0.1

    #     assert type(result_reverse) == DescriptorArray
    #     assert result_reverse.name == result_reverse.unique_name
    #     assert result_reverse.value == -1.0
    #     assert result_reverse.unit == "dimensionless"
    #     assert result_reverse.variance == 0.1

    # @pytest.mark.parametrize("test", [1.0, DescriptorArray("test", 2, "s",)], ids=["sub_array_to_unit", "incompatible_units"])
    # def test_subtraction_exception(self, descriptor: DescriptorArray, test):
    #     # When Then Expect
    #     with pytest.raises(UnitError):
    #         result = test - descriptor
    #     with pytest.raises(UnitError):
    #         result_reverse = descriptor - test

    # @pytest.mark.parametrize("test, expected", [
    #     (DescriptorArray("test", 2, "m", 0.01,),   DescriptorArray("test * name", 2, "m^2", 0.41)),
    #     (DescriptorArray("test", 2, "dm", 0.01),   DescriptorArray("test * name", 0.2, "m^2", 0.0041))],
    #     ids=["regular", "base_unit_conversion"])
    # def test_multiplication(self, descriptor: DescriptorArray, test, expected):
    #     # When Then
    #     result = test * descriptor

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == expected.value
    #     assert result.unit == expected.unit
    #     assert result.variance == pytest.approx(expected.variance)

    # def test_multiplication_with_array(self, descriptor: DescriptorArray):
    #     # When Then
    #     result = descriptor * 2.0
    #     result_reverse = 2.0 * descriptor

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == 2.0
    #     assert result.unit == "m"
    #     assert result.variance == 0.4

    #     assert type(result_reverse) == DescriptorArray
    #     assert result_reverse.name == result_reverse.unique_name
    #     assert result_reverse.value == 2.0
    #     assert result_reverse.unit == "m"
    #     assert result_reverse.variance == 0.4

    # @pytest.mark.parametrize("test, expected, expected_reverse", [
    #     (DescriptorArray("test", 2, "m^2", 0.01,),   DescriptorArray("name / test", 0.5, "1/m", 0.025625), DescriptorArray("test / name", 2, "m", 0.41)),
    #     (2, DescriptorArray("name / 2", 0.5, "m", 0.025), DescriptorArray("2 / name", 2, "1/m", 0.4))],
    #     ids=["DescriptorArray", "scalar"])
    # def test_division(self, descriptor: DescriptorArray, test, expected, expected_reverse):
    #     # When Then
    #     result = descriptor / test
    #     result_reverse = test / descriptor

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == expected.value
    #     assert result.unit == expected.unit
    #     assert result.variance == pytest.approx(expected.variance)

    #     assert type(result_reverse) == DescriptorArray
    #     assert result_reverse.name == result_reverse.unique_name
    #     assert result_reverse.value == expected_reverse.value
    #     assert result_reverse.unit == expected_reverse.unit
    #     assert result_reverse.variance == pytest.approx(expected_reverse.variance)

    # @pytest.mark.parametrize("test", [0, DescriptorArray("test", 0, "m", 0.01)], ids=["zero", "zero_descriptor"])
    # def test_division_exception(self, descriptor: DescriptorArray, test):
    #     # When Then Expect
    #     with pytest.raises(ZeroDivisionError):
    #         result = descriptor / test

    # def test_division_exception_reverse(self):
    #     # When 
    #     descriptor = DescriptorArray(name="name", value=0, variance=0.1)

    #     # Then Expect
    #     with pytest.raises(ZeroDivisionError):
    #         result = 2 / descriptor

    # @pytest.mark.parametrize("test, expected", [
    #     (DescriptorArray("test", 2), DescriptorArray("name ** test", 4, unit="m^2", variance=1.6)),
    #     (2, DescriptorArray("name ** 2", 4, unit="m^2", variance=1.6)),
    #     (-2, DescriptorArray("name ** -2", 0.25, unit="1/m^2", variance=0.00625))],
    #     ids=["DescriptorArray", "scalar", "negative_array"])
    # def test_power_of_descriptor(self, test, expected):
    #     # When 
    #     descriptor = DescriptorArray(name="name", value=2, unit="m", variance=0.1) 

    #     # Then
    #     result = descriptor ** test

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == expected.value
    #     assert result.unit == expected.unit
    #     assert result.variance == expected.variance

    # def test_power_of_dimensionless_descriptor(self):
    #     # When 
    #     descriptor = DescriptorArray(name="name", value=2, unit="dimensionless", variance=0.1) 

    #     # Then
    #     result = descriptor ** 0.5

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == 1.4142135623730951
    #     assert result.unit == "dimensionless"
    #     assert result.variance == pytest.approx(0.0125)

    # @pytest.mark.parametrize("descriptor, exponent, exception", [
    #     (DescriptorArray("name", 2), DescriptorArray("test", 2, unit="m"), UnitError),
    #     (DescriptorArray("name", 2), DescriptorArray("test", 2, variance=0.1), ValueError),
    #     (DescriptorArray("name", 2, unit="m"), 0.5, UnitError),
    #     (DescriptorArray("name", -2), 0.5, ValueError)],
    #     ids=["descriptor_unit", "descriptor_variance", "fractional_of_unit", "fractonal_of_negative"])
    # def test_power_of_descriptor_exceptions(self, descriptor, exponent, exception):
    #     # When Then Expect
    #     with pytest.raises(exception):
    #         result = descriptor ** exponent


    # def test_descriptor_as_exponentiation(self):
    #     # When 
    #     descriptor = DescriptorArray(name="name", value=2) 

    #     # Then
    #     result = 2 ** descriptor

    #     # Expect
    #     assert result == 4

    # @pytest.mark.parametrize("exponent, exception", [
    #     (DescriptorArray("test", 2, unit="m"), UnitError),
    #     (DescriptorArray("test", 2, variance=0.1), ValueError)],
    #     ids=["descriptor_unit", "descriptor_variance"])
    # def test_descriptor_as_exponentiation_exception(self, exponent, exception):
    #     # When Then Expect
    #     with pytest.raises(exception):
    #         result = 2 ** exponent

    # def test_negation(self):
    #     # When 
    #     descriptor = DescriptorArray(name="name", unit="m", value=2, variance=0.1) 

    #     # Then
    #     result = -descriptor

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == -2
    #     assert result.unit == "m"
    #     assert result.variance == 0.1

    # def test_abs(self):
    #     # When 
    #     descriptor = DescriptorArray(name="name", unit="m", value=-2, variance=0.1) 

    #     # Then
    #     result = abs(descriptor)

    #     # Expect
    #     assert type(result) == DescriptorArray
    #     assert result.name == result.unique_name
    #     assert result.value == 2
    #     assert result.unit == "m"
    #     assert result.variance == 0.1

