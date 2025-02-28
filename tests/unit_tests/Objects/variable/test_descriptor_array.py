import pytest
from unittest.mock import MagicMock
import scipp as sc
from scipp import UnitError
from scipp.testing import assert_identical

import numpy as np

from easyscience.Objects.variable.descriptor_array import DescriptorArray
from easyscience.Objects.variable.descriptor_number import DescriptorNumber
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
    def descriptor_dimensionless(self):
        descriptor = DescriptorArray(
            name="name",
            value=[[1., 2.], [3., 4.], [5., 6.]],
            unit="dimensionless",
            variance=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
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
        other = sc.array(dims=('row','column'), 
                         values=[[1.0, 2.0], [3.0, 4.0]], 
                         unit='m', 
                         variances=[[0.1, 0.2], [0.3, 0.4]])
        assert_identical(descriptor.full_value, other)
        
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
        # Expect
        assert np.allclose(descriptor.error, np.sqrt(np.array([[0.2, 0.3], [0.4, 0.5]])))
        assert np.allclose(descriptor.variance, np.array([[0.2, 0.3], [0.4, 0.5]]))


    def test_value(self, descriptor: DescriptorArray):
        # When Then Expect
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

    @pytest.mark.parametrize("test, expected, raises_warning", [
        (DescriptorNumber("test", 2, "m", 0.01),
         DescriptorArray("test + name", 
                         [[3.0, 4.0], [5.0, 6.0]], 
                         "m", 
                         [[0.11, 0.21], [0.31, 0.41]]),
         True),
        (DescriptorNumber("test", 1, "cm", 10),
         DescriptorArray("test + name", 
                         [[101.0, 201.0], [301.0, 401.0]], 
                         "cm", 
                         [[1010.0, 2010.0], [3010.0, 4010.0]]),
         True),
        (DescriptorArray("test", 
                         [[2.0, 3.0], [4.0, -5.0]], 
                         "cm", 
                         [[1.0, 2.0], [3.0, 4.0]]),
         DescriptorArray("test + name", 
                         [[102.0, 203.0], [304.0, 395.0]], 
                         "cm", 
                         [[1001.0, 2002.0], [3003.0, 4004.0]]),
         False)],
        ids=["descriptor_number_regular", "descriptor_number_unit_conversion", "array_conversion"])
    def test_addition(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = test + descriptor
                result_reverse = descriptor + test
            assert len(record) == 2
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = test + descriptor
            result_reverse = descriptor + test
        # Expect
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert result_reverse.unit == descriptor.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'


    @pytest.mark.parametrize("test, expected", [
        ([[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]], 
         DescriptorArray("test", 
                         [[3.0, 5.0], [7.0, -1.0], [11.0, -2.0]], 
                         "dimensionless", 
                         [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])),
        (1,
         DescriptorArray("test", 
                         [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], 
                         "dimensionless", 
                         [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        ],
        ids=["list", "number"])
    def test_addition_dimensionless(self, descriptor_dimensionless: DescriptorArray, test, expected):
        # When Then
        result = test + descriptor_dimensionless
        result_reverse = descriptor_dimensionless + test
        # Expect
        assert type(result) == DescriptorArray
        assert type(result_reverse) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'
        
    @pytest.mark.parametrize("test, expected, raises_warning", [
        (DescriptorNumber("test", 2, "m", 0.01),
         DescriptorArray("test + name", 
                         [[1.0, 0.0], [-1.0, -2.0]], 
                         "m", 
                         [[0.11, 0.21], [0.31, 0.41]]),
         True),
        (DescriptorNumber("test", 1, "cm", 10),
         DescriptorArray("test + name", 
                         [[-99.0, -199.0], [-299.0, -399.0]], 
                         "cm", 
                         [[1010.0, 2010.0], [3010.0, 4010.0]]),
         True),
        (DescriptorArray("test", 
                         [[2.0, 3.0], [4.0, -5.0]], 
                         "cm", 
                         [[1.0, 2.0], [3.0, 4.0]]),
         DescriptorArray("test + name", 
                         [[-98.0, -197.0], [-296.0, -405.0]], 
                         "cm", 
                         [[1001.0, 2002.0], [3003.0, 4004.0]]),
         False)],
        ids=["descriptor_number_regular", "descriptor_number_unit_conversion", "array_conversion"])
    def test_subtraction(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = test - descriptor
                result_reverse = descriptor - test
            assert len(record) == 2
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = test - descriptor
            result_reverse = descriptor - test
        # Expect
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert result_reverse.unit == descriptor.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'
        # Convert units and check that reverse result is the same
        result_reverse.convert_unit(result.unit)
        assert np.array_equal(result.value, -result_reverse.value)

    @pytest.mark.parametrize("test, expected", [
        ([[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]], 
         DescriptorArray("test", 
                         [[1.0, 1.0], [1.0, -9.0], [1.0, -14.0]], 
                         "dimensionless", 
                         [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])),
        (1,
         DescriptorArray("test", 
                         [[0.0, -1.0], [-2.0, -3.0], [-4.0, -5.0]], 
                         "dimensionless", 
                         [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        ],
        ids=["list", "number"])
    def test_subtraction_dimensionless(self, descriptor_dimensionless: DescriptorArray, test, expected):
        # When Then
        result = test - descriptor_dimensionless
        result_reverse = descriptor_dimensionless - test
        # Expect
        assert type(result) == DescriptorArray
        assert type(result_reverse) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.array_equal(result.value, -result_reverse.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'
        

    @pytest.mark.parametrize("test, expected, raises_warning", [
        (DescriptorNumber("test", 2, "m", 0.01),
         DescriptorArray("test * name", 
                         [[2.0, 4.0], [6.0, 8.0]], 
                         "m^2", 
                         [[0.41, 0.84], [1.29, 1.76]]),
         True),
        (DescriptorNumber("test", 1, "cm", 10),
         DescriptorArray("test * name", 
                         [[100.0, 200.0], [300.0, 400.0]], 
                         "cm^2", 
                         [[101000.0, 402000.0], [903000.0, 1604000.0]]),
         True),
        (DescriptorNumber("test", 1, "kg", 10),
         DescriptorArray("test * name", 
                         [[1.0, 2.0], [3.0, 4.0]], 
                         "kg*m", 
                         [[10.1, 40.2], [90.3, 160.4]]),
         True),
        (DescriptorArray("test", 
                         [[2.0, 3.0], [4.0, -5.0]], 
                         "cm", 
                         [[1.0, 2.0], [3.0, 4.0]]),
         DescriptorArray("test * name", 
                         [[200.0, 600.0], [1200.0, -2000.0]], 
                         "cm^2", 
                         [[14000.0, 98000.0], [318000.0, 740000.0]]),
         False),
        ([[2.0, 3.0], [4.0, -5.0]], 
         DescriptorArray("test * name", 
                         [[2.0, 6.0], [12.0, -20.0]], 
                         "m", 
                         [[0.1 * 2**2, 0.2 * 3**2],
                          [0.3 * 4**2, 0.4 * 5**2]]),
         False),
        (2.0, 
         DescriptorArray("test * name", 
                         [[2.0, 4.0], [6.0, 8.0]], 
                         "m", 
                         [[0.1 * 2**2, 0.2 * 2**2],
                          [0.3 * 2**2, 0.4 * 2**2]]),
         False)

        ],
        ids=["descriptor_number_regular",
             "descriptor_number_unit_conversion",
             "descriptor_number_different_units",
             "array_conversion",
             "list",
             "number"])
    def test_multiplication(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = test * descriptor
                result_reverse = descriptor * test
            assert len(record) == 2
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = test * descriptor
            result_reverse = descriptor * test
        # Expect
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'


    @pytest.mark.parametrize("test, expected", [
        ([[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]], 
         DescriptorArray("test", 
                         [[2.0, 6.0], [12.0, -20.0], [30.0, -48.0]], 
                         "dimensionless", 
                         [[0.4, 1.8], [4.8, 10.0], [18.0, 38.4]])),
        (1.5,
         DescriptorArray("test", 
                         [[1.5, 3.0], [4.5, 6.0], [7.5, 9.0]], 
                         "dimensionless", 
                         [[0.225, 0.45], [0.675, 0.9], [1.125, 1.35]]))
        ],
        ids=["list", "number"])
    def test_multiplication_dimensionless(self, descriptor_dimensionless: DescriptorArray, test, expected):
        # When Then
        result = test * descriptor_dimensionless
        result_reverse = descriptor_dimensionless * test
        # Expect
        assert type(result) == DescriptorArray
        assert type(result_reverse) == DescriptorArray
        assert np.array_equal(result.value, expected.value)
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'
    
    @pytest.mark.parametrize("test, expected, raises_warning", [
        (DescriptorNumber("test", 2, "m", 0.01),
         DescriptorArray("test / name", 
                         [[2.0, 1.0], [2.0/3.0, 0.5]], 
                         "dimensionless", 
                         [[0.41, 0.0525], 
                          [(0.01 + 0.3 * 2**2 / 3.0**2) / 3.0**2,
                           (0.01 + 0.4 * 2**2 / 4.0**2) / 4.0**2]]),
         True),
        (DescriptorNumber("test", 1, "cm", 10),
         DescriptorArray("test / name", 
                         [[1.0/100.0, 1.0/200.0], [1.0/300.0, 1.0/400.0]], 
                         "dimensionless", 
                         [[1.01e-3, (1e-3 + 0.2 * 0.01**2/2**2) / 2**2],
                          [(1e-3 + 0.3 * 0.01**2/3**2) / 3**2,(1e-3 + 0.4 * 0.01**2 / 4**2) / 4**2]]),
         True),
        (DescriptorNumber("test", 1, "kg", 10),
         DescriptorArray("test / name", 
                         [[1.0, 0.5], [1.0/3.0, 0.25]], 
                         "kg/m", 
                         [[10.1, ( 10 + 0.2 * 1/2**2 ) / 2**2],
                          [( 10 + 0.3 * 1/3**2 ) / 3**2, ( 10 + 0.4 * 1/4**2 ) / 4**2 ]]),
         True),
        (DescriptorArray("test", 
                         [[2.0, 3.0], [4.0, -5.0]], 
                         "cm^2", 
                         [[1.0, 2.0], [3.0, 4.0]]),
         DescriptorArray("test / name", 
                         [[2e-4, 1.5e-4], [4.0/3.0*1e-4, -1.25e-4]], 
                         "m", 
                         [[1.4e-8, 6.125e-9], 
                          [( 3.0e-8 + 0.3 * (0.0004)**2 / 3**2 ) / 3**2, 
                           ( 4.0e-8 + 0.4 * (0.0005)**2 / 4**2 ) / 4**2]]),
         False),
        ([[2.0, 3.0], [4.0, -5.0]],
         DescriptorArray("test / name", 
                         [[2, 1.5], [4.0/3.0, -1.25]], 
                         "1/m", 
                         [[0.1 * 2**2 / 1**4, 0.2 * 3.0**2 / 2.0**4], 
                          [0.3 * 4**2 / 3**4, 0.4 * 5.0**2 / 4**4]]),
         False),
        (2.0,
         DescriptorArray("test / name", 
                         [[2, 1.0], [2.0/3.0, 0.5]], 
                         "1/m", 
                         [[0.1 * 2**2 / 1**4, 0.2 * 2.0**2 / 2.0**4], 
                          [0.3 * 2**2 / 3**4, 0.4 * 2.0**2 / 4.0**4]]),
         False)
        ],
        ids=["descriptor_number_regular",
             "descriptor_number_unit_conversion",
             "descriptor_number_different_units",
             "array_conversion",
             "list",
             "number"])
    def test_division(self, descriptor: DescriptorArray, test, expected, raises_warning):
        # When Then
        if raises_warning:
            with pytest.warns(UserWarning) as record:
                result = test / descriptor
                result_reverse = descriptor / test
            assert len(record) == 2
            assert 'Correlations introduced' in record[0].message.args[0]
        else:
            result = test / descriptor
            result_reverse = descriptor / test
        # Expect
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.allclose(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize("test, expected", [
        ([[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]], 
         DescriptorArray("test", 
                         [[2.0/1.0, 3.0/2.0], [4.0/3.0, -5.0/4.0], [6.0/5.0, -8.0/6.0]], 
                         "dimensionless", 
                         [[0.1 * 2.0**2, 0.2 * 3.0**2 / 2**4],
                          [0.3 * 4.0**2 / 3.0**4, 0.4 * 5.0**2 / 4**4],
                          [0.5 * 6.0**2 / 5**4, 0.6 * 8.0**2 / 6**4]])),
        (2,
         DescriptorArray("test", 
                         [[2.0, 1.0], [2.0/3.0, 0.5], [2.0/5.0, 1.0/3.0]], 
                         "dimensionless", 
                         [[0.1 * 2.0**2, 0.2 / 2**2],
                          [0.3 * 2**2 / 3**4, 0.4 * 2**2 / 4**4], 
                          [0.5 * 2**2 / 5**4, 0.6 * 2**2 / 6**4]]))
        ],
        ids=["list", "number"])
    def test_division_dimensionless(self, descriptor_dimensionless: DescriptorArray, test, expected):
        # When Then
        result = test / descriptor_dimensionless
        result_reverse = descriptor_dimensionless / test
        # Expect
        assert type(result) == DescriptorArray
        assert type(result_reverse) == DescriptorArray
        assert np.allclose(result.value, expected.value)
        assert np.allclose(result.value, 1 / result_reverse.value)
        assert np.allclose(result.variance, expected.variance)

        assert descriptor_dimensionless.unit == 'dimensionless'
    
    @pytest.mark.parametrize("test", [
        [[2.0, 3.0], [4.0, -5.0], [6.0, 0.0]], 
        0.0,
        DescriptorNumber("test", 0, "cm", 10),
        DescriptorArray("test", 
                        [[1.5, 0.0], [4.5, 6.0], [7.5, 9.0]], 
                        "dimensionless", 
                        [[0.225, 0.45], [0.675, 0.9], [1.125, 1.35]])],
        ids=["list", "number", "DescriptorNumber", "DescriptorArray"])
    def test_division_exception(self, descriptor_dimensionless: DescriptorArray, test):
        # When Then
        with pytest.raises(ZeroDivisionError):
            descriptor_dimensionless / test
        
        # Also test reverse division where `self` is a DescriptorArray with a zero
        zero_descriptor = DescriptorArray("test", 
                                          [[1.5, 0.0], [4.5, 6.0], [7.5, 0.0]], 
                                          "dimensionless", 
                                          [[0.225, 0.45], [0.675, 0.9], [1.125, 1.35]])
        with pytest.raises(ZeroDivisionError):
            test / zero_descriptor
        
    @pytest.mark.parametrize("test, expected", [
        (DescriptorNumber("test", 2, "dimensionless"),
         DescriptorArray("test ** name", 
                         [[1.0, 4.0], [9.0, 16.0]], 
                         "m^2", 
                         [[4 * 0.1 * 1, 4 * 0.2 * 2**2],
                          [4 * 0.3 * 3**2, 4 * 0.4 * 4**2]])),
        (DescriptorNumber("test", 3, "dimensionless"),
         DescriptorArray("test ** name", 
                         [[1.0, 8.0], [27, 64.0]], 
                         "m^3", 
                         [[9 * 0.1, 9 * 0.2 * 2**4],
                          [9 * 0.3 * 3**4, 9 * 0.4 * 4**4]])),
        (DescriptorNumber("test", 0.0, "dimensionless"),
         DescriptorArray("test ** name", 
                         [[1.0, 1.0], [1.0, 1.0]], 
                         "dimensionless", 
                         [[0.0, 0.0], [0.0, 0.0]])),
        (0.0,
         DescriptorArray("test ** name", 
                         [[1.0, 1.0], [1.0, 1.0]], 
                         "dimensionless", 
                         [[0.0, 0.0], [0.0, 0.0]]))
         ],
        ids=["descriptor_number_squared",
             "descriptor_number_cubed",
             "descriptor_number_zero",
             "number_zero"])
    def test_power(self, descriptor: DescriptorArray, test, expected):
        # When Then
        result = descriptor ** test
        # Expect
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'

    @pytest.mark.parametrize("test, expected", [
        (DescriptorNumber("test", 0.1, "dimensionless"),
         DescriptorArray("test ** name", 
                         [[1, 2**0.1], [3**0.1, 4**0.1], [5**0.1, 6**0.1]], 
                         "dimensionless", 
                         [[0.1**2 * 0.1 * 1, 0.1**2 * 0.2 * 2**(-1.8)],
                          [0.1**2 * 0.3 * 3**(-1.8), 0.1**2 * 0.4 * 4**(-1.8)],
                          [0.1**2 * 0.5 * 5**(-1.8), 0.1**2 * 0.6 * 6**(-1.8)]])),
        (DescriptorNumber("test", 2.0, "dimensionless"),
         DescriptorArray("test ** name", 
                         [[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], 
                         "dimensionless", 
                         [[0.4, 3.2], [10.8, 25.6], [50., 86.4]])),
        ],
        ids=["descriptor_number_fractional", "descriptor_number_integer"])
    def test_power_dimensionless(self, descriptor_dimensionless: DescriptorArray, test, expected):
        # When Then
        result = descriptor_dimensionless ** test
        # Expect
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.allclose(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor_dimensionless.unit == 'dimensionless'

    
    @pytest.mark.parametrize("test, exception", [
        (DescriptorNumber("test", 2, "m"), UnitError),
        (DescriptorNumber("test", 2, "dimensionless", 10), ValueError),
        (DescriptorNumber("test", np.nan, "dimensionless"), UnitError),
        (DescriptorNumber("test", np.nan, "dimensionless"), UnitError),
        (DescriptorNumber("test", 1.5, "dimensionless"), UnitError),
        (DescriptorNumber("test", 0.5, "dimensionless"), UnitError)  # Square roots are not legal
        ],
        ids=["units",
             "variance",
             "scipp_nan",
             "nan_result",
             "non_integer_exponent_on_units",
             "square_root_on_units"
             ])
    def test_power_exception(self, descriptor: DescriptorArray, test, exception):
        # When Then
        with pytest.raises(exception):
            result = descriptor ** 2 ** test
        with pytest.raises(TypeError):
            # Exponentiation with an array does not make sense
            test ** descriptor  

    @pytest.mark.parametrize("test", [
        DescriptorNumber("test", 2, "s"),
        DescriptorArray("test", [[1, 2], [3, 4]], "s")], ids=["add_array_to_unit", "incompatible_units"])
    def test_operation_exception(self, descriptor: DescriptorArray, test):
        # When Then Expect
        import operator
        operators = [operator.add,
                     operator.sub]
        for operator in operators:  # This can probably be done better w. fixture
            with pytest.raises(UnitError):
                result = operator(descriptor, test)
            with pytest.raises(UnitError):
                result_reverse = operator(test, descriptor)
    
    @pytest.mark.parametrize("function", [
            np.sin,
            np.cos,
            np.exp,
            np.add,
            np.multiply
        ],
        ids=["sin", "cos", "exp", "add", "multiply"])
    def test_numpy_ufuncs_exception(self, descriptor_dimensionless, function):
        (np.add,np.array([[2.0, 3.0], [4.0, -5.0], [6.0, -8.0]])),
        """
        Not implemented ufuncs should return NotImplemented.
        """
        test = np.array([[1, 2], [3, 4]])
        with pytest.raises(TypeError) as e:
            function(descriptor_dimensionless, test)
        assert 'returned NotImplemented from' in str(e)
    
    def test_negation(self, descriptor):
        # When 
        # Then
        result = -descriptor

        # Expect
        expected = DescriptorArray(
            name="name",
            value=[[-1., -2.], [-3., -4.]],
            unit="m",
            variance=[[0.1, 0.2], [0.3, 0.4]],
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.array_equal(result.value, expected.value)
        assert result.unit == expected.unit
        assert np.allclose(result.variance, expected.variance)
        assert descriptor.unit == 'm'


    def test_abs(self, descriptor):
        # When 
        negated = DescriptorArray(
            name="name",
            value=[[-1., -2.], [-3., -4.]],
            unit="m",
            variance=[[0.1, 0.2], [0.3, 0.4]],
            description="description",
            url="url",
            display_name="display_name",
            parent=None,
        )

        # Then
        result = abs(negated)

        # Expect
        assert type(result) == DescriptorArray
        assert result.name == result.unique_name
        assert np.array_equal(result.value, descriptor.value)
        assert result.unit == descriptor.unit
        assert np.allclose(result.variance, descriptor.variance)
        assert descriptor.unit == 'm'

