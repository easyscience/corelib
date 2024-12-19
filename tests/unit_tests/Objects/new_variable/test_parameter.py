import pytest
from unittest.mock import MagicMock
import scipp as sc
import numpy as np

from scipp import UnitError

from easyscience.Objects.new_variable.parameter import Parameter
from easyscience.Objects.new_variable.descriptor_number import DescriptorNumber
from easyscience import global_object

class TestParameter:
    @pytest.fixture
    def parameter(self) -> Parameter:
        self.mock_callback = MagicMock()
        parameter = Parameter(
            name="name",
            value=1,
            unit="m",
            variance=0.01,
            min=0,
            max=10,
            description="description",
            url="url",
            display_name="display_name",
            callback=self.mock_callback,
            enabled="enabled",
            parent=None,
        )
        return parameter

    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def test_init(self, parameter: Parameter):
        # When Then Expect
        assert parameter._min.value == 0
        assert parameter._min.unit == "m"
        assert parameter._max.value == 10
        assert parameter._max.unit == "m"
        assert parameter._callback == self.mock_callback
        assert parameter._enabled == "enabled"

        # From super
        assert parameter._scalar.value == 1
        assert parameter._scalar.unit == "m"
        assert parameter._scalar.variance == 0.01
        assert parameter._name == "name"
        assert parameter._description == "description"
        assert parameter._url == "url"
        assert parameter._display_name == "display_name"

    def test_init_value_min_exception(self):
        # When 
        mock_callback = MagicMock()
        value = -1

        # Then Expect
        with pytest.raises(ValueError):
            Parameter(
                name="name",
                value=value,
                unit="m",
                variance=0.01,
                min=0,
                max=10,
                description="description",
                url="url",
                display_name="display_name",
                callback=mock_callback,
                enabled="enabled",
                parent=None,
            )

    def test_init_value_max_exception(self):
        # When 
        mock_callback = MagicMock()
        value = 100

        # Then Expect
        with pytest.raises(ValueError):
            Parameter(
                name="name",
                value=value,
                unit="m",
                variance=0.01,
                min=0,
                max=10,
                description="description",
                url="url",
                display_name="display_name",
                callback=mock_callback,
                enabled="enabled",
                parent=None,
            )

    def test_min(self, parameter: Parameter):
        # When Then Expect
        assert parameter.min == 0

    def test_set_min(self, parameter: Parameter):
        # When Then 
        self.mock_callback.fget.return_value = 1.0  # Ensure fget returns a scalar value

        parameter.min = 0.1

        # Expect
        assert parameter.min == 0.1

    def test_set_min_exception(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.min = 10

    def test_set_max(self, parameter: Parameter):
        # When Then 
        parameter.max = 10

        # Expect
        assert parameter.max == 10

    def test_set_max_exception(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.max = 0.1

    def test_convert_unit(self, parameter: Parameter):
        # When Then
        parameter.convert_unit("mm")

        # Expect
        assert parameter._min.value == 0
        assert parameter._min.unit == "mm"
        assert parameter._max.value == 10000
        assert parameter._max.unit == "mm"

    def test_fixed(self, parameter: Parameter):
        # When Then Expect
        assert parameter.fixed == False

    def test_set_fixed(self, parameter: Parameter):
        # When Then 
        parameter.fixed = True

        # Expect
        assert parameter.fixed == True

    @pytest.mark.parametrize("fixed", ["True", 1])
    def test_set_fixed_exception(self, parameter: Parameter, fixed):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.fixed = fixed

    def test_error(self, parameter: Parameter):
        # When Then Expect
        assert parameter.error == 0.1
    
    def test_set_error(self, parameter: Parameter):
        # When 
        parameter.error = 10

        # Then Expect
        assert parameter.error == 10
        assert parameter._scalar.variance == 100

    def test_set_error_exception(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.error = -0.1

    # Commented out because __float__ method might be removed
    # def test_float(self, parameter: Parameter):
    #     # When Then Expect
    #     assert float(parameter) == 1.0

    def test_repr(self, parameter: Parameter):
        # When Then Expect
        assert repr(parameter) == "<Parameter 'name': 1.0000 ± 0.1000 m, bounds=[0.0:10.0]>"

    def test_repr_fixed(self, parameter: Parameter):
        # When 
        parameter.fixed = True

        # Then Expect
        assert repr(parameter) == "<Parameter 'name': 1.0000 ± 0.1000 m (fixed), bounds=[0.0:10.0]>"

    def test_bounds(self, parameter: Parameter):
        # When Then Expect
        assert parameter.bounds == (0, 10)
    
    def test_set_bounds(self, parameter: Parameter):
        # When 
        self.mock_callback.fget.return_value = 1.0  # Ensure fget returns a scalar value
        parameter._enabled = False
        parameter._fixed = True

        # Then 
        parameter.bounds = (-10, 5)

        # Expect
        assert parameter.min == -10
        assert parameter.max == 5
        assert parameter._enabled == True
        assert parameter._fixed == False

    def test_set_bounds_exception_min(self, parameter: Parameter):
        # When 
        parameter._enabled = False
        parameter._fixed = True

        # Then
        with pytest.raises(ValueError):
            parameter.bounds = (2, 10)

        # Expect
        assert parameter.min == 0
        assert parameter.max == 10
        assert parameter._enabled == False
        assert parameter._fixed == True

    def test_set_bounds_exception_max(self, parameter: Parameter):
        # When 
        parameter._enabled = False
        parameter._fixed = True

        # Then
        with pytest.raises(ValueError):
            parameter.bounds = (0, 0.1)

        # Expect
        assert parameter.min == 0
        assert parameter.max == 10
        assert parameter._enabled == False
        assert parameter._fixed == True

    def test_enabled(self, parameter: Parameter):
        # When
        parameter._enabled = True
        
        # Then Expect
        assert parameter.enabled is True

    def test_set_enabled(self, parameter: Parameter):
        # When
        parameter.enabled = False

        # Then Expect
        assert parameter._enabled is False

    def test_value_match_callback(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = 1.0

        # Then Expect
        assert parameter.value == 1.0
        assert parameter._callback.fget.call_count == 1
        
    def test_value_no_match_callback(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = 2.0

        # Then Expect
        assert parameter.value == 2.0
        assert parameter._callback.fget.call_count == 1

    def test_set_value(self, parameter: Parameter):
        # When
        # First call returns 1.0 that is used to enforce the undo/redo functionality to register the value has changed
        # Second and third call returns 2.0 is used in the constraint check
        self.mock_callback.fget.side_effect = [1.0, 2.0, 2.0]

        # Then
        parameter.value = 2

        # Expect
        parameter._callback.fset.assert_called_with(2)
        assert parameter._callback.fset.call_count == 1
        assert parameter._scalar == sc.scalar(2, unit='m')

    def test_full_value_match_callback(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = sc.scalar(1, unit='m')

        # Then Expect
        assert parameter.full_value == sc.scalar(1, unit='m')
        assert parameter._callback.fget.call_count == 1
        
    def test_full_value_no_match_callback(self, parameter: Parameter):
        # When
        self.mock_callback.fget.return_value = sc.scalar(2, unit='m')

        # Then Expect
        assert parameter.full_value == sc.scalar(2, unit='m')
        assert parameter._callback.fget.call_count == 1

    def test_set_full_value(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(AttributeError):
            parameter.full_value = sc.scalar(2, unit='s')
    
    def test_copy(self, parameter: Parameter):
        # When Then
        self.mock_callback.fget.return_value = 1.0  # Ensure fget returns a scalar value
        parameter_copy = parameter.__copy__()

        # Expect
        assert type(parameter_copy) == Parameter
        assert id(parameter_copy._scalar) != id(parameter._scalar)
        assert isinstance(parameter_copy._callback, property)

        assert parameter_copy._name == parameter._name
        assert parameter_copy._scalar == parameter._scalar
        assert parameter_copy._min == parameter._min
        assert parameter_copy._max == parameter._max
        assert parameter_copy._fixed == parameter._fixed
        assert parameter_copy._description == parameter._description
        assert parameter_copy._url == parameter._url
        assert parameter_copy._display_name == parameter._display_name
        assert parameter_copy._enabled == parameter._enabled

    def test_as_data_dict(self, clear, parameter: Parameter):
        # When Then
        self.mock_callback.fget.return_value = 1.0  # Ensure fget returns a scalar value
        parameter_dict = parameter.as_data_dict()

        # Expect
        assert parameter_dict == {
            "name": "name",
            "value": 1.0,
            "unit": "m",
            "variance": 0.01,
            "min": 0,
            "max": 10,
            "fixed": False,
            "description": "description",
            "url": "url",
            "display_name": "display_name",
            "enabled": "enabled",
            "unique_name": "Parameter_0",
        }

    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (Parameter("test", 2, "m", 0.01, -10, 20),  Parameter("name + test", 3, "m", 0.02, -10, 30),                Parameter("test + name", 3, "m", 0.02, -10, 30)),
            (Parameter("test", 2, "m", 0.01),           Parameter("name + test", 3, "m", 0.02, min=-np.inf, max=np.inf),Parameter("test + name", 3, "m", 0.02, min=-np.inf, max=np.inf)),
            (Parameter("test", 2, "cm", 0.01, -10, 10), Parameter("name + test", 1.02, "m", 0.010001, -0.1, 10.1),      Parameter("test + name", 102, "cm", 100.01, -10, 1010))],
            ids=["regular", "no_bounds", "unit_conversion"])
    def test_addition_with_parameter(self, parameter : Parameter, test : Parameter, expected : Parameter, expected_reverse : Parameter):
        # When 
        parameter._callback = property()

        # Then
        result = parameter + test
        result_reverse = test + parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

        assert result_reverse.name == result_reverse.unique_name 
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == expected_reverse.variance
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

        assert parameter.unit == "m"

    def test_addition_with_scalar(self):
        # When
        parameter = Parameter(name="name", value=1, variance=0.01, min=0, max=10)

        # Then
        result = parameter + 1.0
        result_reverse = 1.0 + parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == 2.0
        assert result.unit == "dimensionless"
        assert result.variance == 0.01
        assert result.min == 1.0
        assert result.max == 11.0

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == 2.0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.01
        assert result_reverse.min == 1.0
        assert result_reverse.max == 11.0

    def test_addition_with_descriptor_number(self, parameter : Parameter):
        # When 
        parameter._callback = property()
        descriptor_number = DescriptorNumber(name="test", value=1, variance=0.1, unit="cm")

        # Then
        result = parameter + descriptor_number
        result_reverse = descriptor_number + parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == result.unique_name
        assert result.value == 1.01
        assert result.unit == "m"
        assert result.variance == 0.01001
        assert result.min == 0.01
        assert result.max == 10.01

        assert type(result_reverse) == Parameter
        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == 101.0
        assert result_reverse.unit == "cm"
        assert result_reverse.variance == 100.1
        assert result_reverse.min == 1
        assert result_reverse.max == 1001

        assert parameter.unit == "m"
        assert descriptor_number.unit == "cm"

    @pytest.mark.parametrize("test", [1.0, Parameter("test", 2, "s",)], ids=["add_scalar_to_unit", "incompatible_units"])
    def test_addition_exception(self, parameter : Parameter, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = parameter + test
        with pytest.raises(UnitError):
            result_reverse = test + parameter
        
    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (Parameter("test", 2, "m", 0.01, -20, 20),  Parameter("name - test", -1, "m", 0.02, -20, 30),                Parameter("test - name", 1, "m", 0.02, -30, 20)),
            (Parameter("test", 2, "m", 0.01),           Parameter("name - test", -1, "m", 0.02, min=-np.inf, max=np.inf),Parameter("test - name", 1, "m", 0.02, min=-np.inf, max=np.inf)),
            (Parameter("test", 2, "cm", 0.01, -10, 10), Parameter("name - test", 0.98, "m", 0.010001, -0.1, 10.1),       Parameter("test - name", -98, "cm", 100.01, -1010, 10))],
            ids=["regular", "no_bounds", "unit_conversion"])
    def test_subtraction_with_parameter(self, parameter : Parameter, test : Parameter, expected : Parameter, expected_reverse : Parameter):
        # When 
        parameter._callback = property()

        # Then
        result = parameter - test
        result_reverse = test - parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == expected_reverse.variance
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

        assert parameter.unit == "m"

    def test_subtraction_with_parameter_nan_cases(self):
        # When
        parameter = Parameter(name="name", value=1, variance=0.01, min=-np.inf, max=np.inf)
        test = Parameter(name="test", value=2, variance=0.01, min=-np.inf, max=np.inf)

        # Then
        result = parameter - test
        result_reverse = test - parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == -1.0
        assert result.unit == "dimensionless"
        assert result.variance == 0.02
        assert result.min == -np.inf
        assert result.max == np.inf

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == 1.0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.02
        assert result_reverse.min == -np.inf
        assert result_reverse.max == np.inf

    def test_subtraction_with_scalar(self):
        # When
        parameter = Parameter(name="name", value=2, variance=0.01, min=0, max=10)

        # Then
        result = parameter - 1.0
        result_reverse = 1.0 - parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == 1.0
        assert result.unit == "dimensionless"
        assert result.variance == 0.01
        assert result.min == -1.0
        assert result.max == 9.0

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == -1.0
        assert result_reverse.unit == "dimensionless"
        assert result_reverse.variance == 0.01
        assert result_reverse.min == -9.0
        assert result_reverse.max == 1.0

    def test_subtraction_with_descriptor_number(self, parameter : Parameter):
        # When 
        parameter._callback = property()
        descriptor_number = DescriptorNumber(name="test", value=1, variance=0.1, unit="cm")

        # Then
        result = parameter - descriptor_number
        result_reverse = descriptor_number - parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == result.unique_name
        assert result.value == 0.99
        assert result.unit == "m"
        assert result.variance == 0.01001
        assert result.min == -0.01
        assert result.max == 9.99

        assert type(result_reverse) == Parameter
        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == -99.0
        assert result_reverse.unit == "cm"
        assert result_reverse.variance == 100.1
        assert result_reverse.min == -999
        assert result_reverse.max == 1

        assert parameter.unit == "m"
        assert descriptor_number.unit == "cm"

    @pytest.mark.parametrize("test", [1.0, Parameter("test", 2, "s",)], ids=["sub_scalar_to_unit", "incompatible_units"])
    def test_subtraction_exception(self, parameter : Parameter, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = parameter - test
        with pytest.raises(UnitError):
            result_reverse = test - parameter

    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (Parameter("test", 2, "m", 0.01, -10, 20),     Parameter("name * test", 2, "m^2", 0.05, -100, 200),               Parameter("test * name", 2, "m^2", 0.05, -100, 200)),
            (Parameter("test", 2, "m", 0.01),              Parameter("name * test", 2, "m^2", 0.05, min=-np.inf, max=np.inf), Parameter("test * name", 2, "m^2", 0.05, min=-np.inf, max=np.inf)),
            (Parameter("test", 2, "dm", 0.01, -10, 20),    Parameter("name * test", 0.2, "m^2", 0.0005, -10, 20),             Parameter("test * name", 0.2, "m^2", 0.0005, -10, 20))],
            ids=["regular", "no_bounds", "base_unit_conversion"])
    def test_multiplication_with_parameter(self, parameter : Parameter, test : Parameter, expected : Parameter, expected_reverse : Parameter):
        # When 
        parameter._callback = property()

        # Then
        result = parameter * test
        result_reverse = test * parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == pytest.approx(expected.variance)
        assert result.min == expected.min
        assert result.max == expected.max

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == pytest.approx(expected_reverse.variance)
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (Parameter("test", 0, "", 0.01, -10, 0),    Parameter("name * test", 0.0, "dimensionless", 0.01, -np.inf, 0), Parameter("test * name", 0, "dimensionless", 0.01, -np.inf, 0)),
            (Parameter("test", 0, "", 0.01, 0, 10),     Parameter("name * test", 0.0, "dimensionless", 0.01, 0, np.inf),  Parameter("test * name", 0, "dimensionless", 0.01, 0, np.inf))],
            ids=["zero_min", "zero_max"])
    def test_multiplication_with_parameter_nan_cases(self, test, expected, expected_reverse):
        # When
        parameter = Parameter(name="name", value=1, variance=0.01, min=1, max=np.inf)

        # Then
        result = parameter * test
        result_reverse = test * parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == expected_reverse.variance
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

    @pytest.mark.parametrize("test, expected, expected_reverse", [
        (DescriptorNumber(name="test", value=2, variance=0.1, unit="cm"), Parameter("name * test", 2, "dm^2", 0.14, 0, 20), Parameter("test * name", 2, "dm^2", 0.14, 0, 20)),
        (DescriptorNumber(name="test", value=0, variance=0.1, unit="cm"), DescriptorNumber("name * test", 0, "dm^2", 0.1), DescriptorNumber("test * name", 0, "dm^2", 0.1))],
        ids=["regular", "zero_value"])
    def test_multiplication_with_descriptor_number(self, parameter : Parameter, test, expected, expected_reverse):
        # When 
        parameter._callback = property()

        # Then
        result = parameter * test
        result_reverse = test * parameter

        # Expect
        assert type(result) == type(expected)
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        if isinstance(result, Parameter):
            assert result.min == expected.min
            assert result.max == expected.max

        assert type(result_reverse) == type(expected_reverse)
        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == expected_reverse.variance
        if isinstance(result_reverse, Parameter):
            assert result_reverse.min == expected_reverse.min
            assert result_reverse.max == expected_reverse.max

    @pytest.mark.parametrize("test, expected, expected_reverse", [
        (2, Parameter("name * 2", 2, "m", 0.04, 0, 20), Parameter("2 * name", 2, "m", 0.04, 0, 20)),
        (0, DescriptorNumber("name * 0", 0, "m", 0), DescriptorNumber("0 * name", 0, "m", 0))],
        ids=["regular", "zero_value"])
    def test_multiplication_with_scalar(self, parameter : Parameter, test, expected, expected_reverse):
        # When 
        parameter._callback = property()

        # Then
        result = parameter * test
        result_reverse = test * parameter

        # Expect
        assert type(result) == type(expected)
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        if isinstance(result, Parameter):
            assert result.min == expected.min
            assert result.max == expected.max

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == expected_reverse.variance
        if isinstance(result_reverse, Parameter):
            assert result_reverse.min == expected_reverse.min
            assert result_reverse.max == expected_reverse.max

    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (Parameter("test", 2, "s", 0.01, -10, 20),  Parameter("name / test", 0.5, "m/s", 0.003125, -np.inf, np.inf),       Parameter("test / name", 2, "s/m", 0.05, -np.inf, np.inf)),
            (Parameter("test", 2, "s", 0.01, 0, 20),    Parameter("name / test", 0.5, "m/s", 0.003125, 0.0, np.inf),           Parameter("test / name", 2, "s/m", 0.05, 0.0, np.inf)),
            (Parameter("test", -2, "s", 0.01, -10, 0),  Parameter("name / test", -0.5, "m/s", 0.003125, -np.inf, 0.0),         Parameter("test / name", -2, "s/m", 0.05, -np.inf, 0.0))],
            ids=["crossing_zero", "only_positive", "only_negative"])
    def test_division_with_parameter(self, parameter : Parameter, test, expected, expected_reverse):
        # When 
        parameter._callback = property()

        # Then
        result = parameter / test
        result_reverse = test / parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == result.unique_name
        assert result.value == pytest.approx(expected.value)
        assert result.unit == expected.unit
        assert result.variance == pytest.approx(expected.variance)
        assert result.min == expected.min
        assert result.max == expected.max

        assert type(result) == Parameter
        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == pytest.approx(expected_reverse.value)
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == pytest.approx(expected_reverse.variance)
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

    @pytest.mark.parametrize("first, second, expected", [
        (Parameter("name", 1, "m", 0.01, -10, 20),    Parameter("test", -2, "s", 0.01, -10, 0),    Parameter("name / test", -0.5, "m/s", 0.003125, -np.inf, np.inf)),
        (Parameter("name", -10, "m", 0.01, -20, -10), Parameter("test", -2, "s", 0.01, -10, 0),    Parameter("name / test", 5.0, "m/s", 0.065, 1, np.inf)),
        (Parameter("name", 10, "m", 0.01, 10, 20),    Parameter("test", -20, "s", 0.01, -20, -10), Parameter("name / test", -0.5, "m/s", 3.125e-5, -2, -0.5))],
        ids=["first_crossing_zero_second_negative_0", "both_negative_second_negative_0", "finite_limits"])
    def test_division_with_parameter_remaining_cases(self, first, second, expected):
        # When Then
        result = first / second

        # Expect
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

    @pytest.mark.parametrize("test, expected, expected_reverse", [
            (DescriptorNumber(name="test", value=2, variance=0.1, unit="s"), Parameter("name / test", 0.5, "m/s", 0.00875, 0, 5), Parameter("test / name", 2, "s/m", 0.14, 0.2, np.inf)),
            (2, Parameter("name / 2", 0.5, "m", 0.0025, 0, 5), Parameter("2 / name", 2, "m**-1", 0.04, 0.2, np.inf))],
            ids=["descriptor_number", "number"])
    def test_division_with_descriptor_number_and_number(self, parameter : Parameter, test, expected, expected_reverse):
        # When 
        parameter._callback = property()

        # Then
        result = parameter / test
        result_reverse = test / parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

        assert type(result_reverse) == Parameter
        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == expected_reverse.value
        assert result_reverse.unit == expected_reverse.unit
        assert result_reverse.variance == expected_reverse.variance
        assert result_reverse.min == expected_reverse.min
        assert result_reverse.max == expected_reverse.max

    @pytest.mark.parametrize("test, expected", [
            (DescriptorNumber(name="test", value=0, variance=0.1, unit="s"), DescriptorNumber("test / name", 0.0, "s/m", 0.1)),
            (0, DescriptorNumber("0 / name", 0.0, "1/m", 0.0))],
            ids=["descriptor_number", "number"])
    def test_zero_value_divided_by_parameter(self, parameter : Parameter, test, expected):
        # When 
        parameter._callback = property()

        # Then
        result = test / parameter

        # Expect
        assert type(result) == DescriptorNumber
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance

    @pytest.mark.parametrize("first, second, expected", [
        (DescriptorNumber("name", 1, "m", 0.01),  Parameter("test", 2, "s", 0.1, -10, 10), Parameter("name / test", 0.5, "m/s", 0.00875, -np.inf, np.inf)),
        (DescriptorNumber("name", -1, "m", 0.01), Parameter("test", 2, "s", 0.1, 0, 10),   Parameter("name / test", -0.5, "m/s", 0.00875, -np.inf, -0.1)),
        (DescriptorNumber("name", 1, "m", 0.01),  Parameter("test", -2, "s", 0.1, -10, 0), Parameter("name / test", -0.5, "m/s", 0.00875, -np.inf, -0.1)),
        (DescriptorNumber("name", -1, "m", 0.01), Parameter("test", -2, "s", 0.1, -10, 0), Parameter("name / test", 0.5, "m/s", 0.00875, 0.1, np.inf)),
        (DescriptorNumber("name", 1, "m", 0.01),  Parameter("test", 2, "s", 0.1, 1, 10),   Parameter("name / test", 0.5, "m/s", 0.00875, 0.1, 1))],
        ids=["crossing_zero", "positive_0_with_negative", "negative_0_with_positive", "negative_0_with_negative", "finite_limits"])
    def test_division_with_descriptor_number_missing_cases(self, first, second, expected):
        # When Then
        result = first / second

        # Expect
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

    @pytest.mark.parametrize("test", [0, DescriptorNumber("test", 0, "s", 0.1)], ids=["number", "descriptor_number"])
    def test_divide_parameter_by_zero(self, parameter : Parameter, test):
        # When 
        parameter._callback = property()

        # Then Expect
        with pytest.raises(ZeroDivisionError):
            result = parameter / test

    def test_divide_by_zero_value_parameter(self):
        # When
        descriptor = DescriptorNumber("test", 1, "s", 0.1)
        parameter = Parameter("name", 0, "m", 0.01)

        # Then Expect
        with pytest.raises(ZeroDivisionError):
            result = descriptor / parameter

    @pytest.mark.parametrize("test, expected", [
        (3, Parameter("name ** 3", 125, "m^3", 281.25, -125, 1000)),
        (2, Parameter("name ** 2", 25, "m^2", 5.0, 0, 100)),
        (-1, Parameter("name ** -1", 0.2, "1/m", 8e-5, -np.inf, np.inf)),
        (-2, Parameter("name ** -2", 0.04, "1/m^2", 1.28e-5, 0, np.inf)),
        (0, DescriptorNumber("name ** 0", 1, "dimensionless", 0)),
        (DescriptorNumber("test", 2), Parameter("name ** test", 25, "m^2", 5.0, 0, 100))],
        ids=["power_3", "power_2", "power_-1", "power_-2", "power_0", "power_descriptor_number"])
    def test_power_of_parameter(self, test, expected):
        # When 
        parameter = Parameter("name", 5, "m", 0.05, -5, 10)

        # Then
        result = parameter ** test

        # Expect
        assert type(result) == type(expected)
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        if isinstance(result, Parameter):
            assert result.min == expected.min
            assert result.max == expected.max

    @pytest.mark.parametrize("test, exponent, expected", [
        (Parameter("name", 5, "m", 0.05, 0, 10),    -1, Parameter("name ** -1", 0.2, "1/m", 8e-5, 0.1, np.inf)),
        (Parameter("name", -5, "m", 0.05, -5, 0),   -1, Parameter("name ** -1", -0.2, "1/m", 8e-5, -np.inf, -0.2)),
        (Parameter("name", 5, "m", 0.05, 5, 10),    -1, Parameter("name ** -1", 0.2, "1/m", 8e-5, 0.1, 0.2)),
        (Parameter("name", -5, "m", 0.05, -10, -5), -1, Parameter("name ** -1", -0.2, "1/m", 8e-5, -0.2, -0.1)),
        (Parameter("name", -5, "m", 0.05, -10, -5), -2, Parameter("name ** -2", 0.04, "1/m^2", 1.28e-5, 0.01, 0.04)),
        (Parameter("name", 5, "", 0.1, 1, 10),     0.3, Parameter("name ** 0.3", 1.6206565966927624, "", 0.0009455500095853564, 1, 1.9952623149688795)),
        (Parameter("name", 5, "", 0.1),            0.5, Parameter("name ** 0.5", 2.23606797749979, "", 0.005, 0, np.inf))],
        ids=["0_positive", "negative_0", "both_positive", "both_negative_invert", "both_negative_invert_square", "fractional", "fractional_negative_limit"])
    def test_power_of_diffent_parameters(self, test, exponent, expected):
        # When Then
        result = test ** exponent

        # Expect
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

    @pytest.mark.parametrize("parameter, exponent, expected", [
        (Parameter("name", 5, "m"), DescriptorNumber("test", 2, unit="s"), UnitError),
        (Parameter("name", 5, "m"), DescriptorNumber("test", 2, variance=0.01), ValueError),
        (Parameter("name", 5, "m"), 0.5, UnitError),
        (Parameter("name", -5, ""), 0.5, ValueError),],
        ids=["exponent_unit", "exponent_variance", "exponent_fractional", "negative_base_fractional"])
    def test_power_exceptions(self, parameter, exponent, expected):
        # When Then Expect
        with pytest.raises(expected):
            result = parameter ** exponent

    def test_negation(self):
        # When
        parameter = Parameter("name", 5, "m", 0.05, -5, 10)

        # Then
        result = -parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == -5
        assert result.unit == "m"
        assert result.variance == 0.05
        assert result.min == -10
        assert result.max == 5

    @pytest.mark.parametrize("test, expected", [
        (Parameter("name", -5, "m", 0.05, -10, -5), Parameter("abs(name)", 5, "m", 0.05, 5, 10)),
        (Parameter("name", 5, "m", 0.05, -10, 10), Parameter("abs(name)", 5, "m", 0.05, 0, 10))],
        ids=["pure_negative", "crossing_zero"])
    def test_abs(self, test, expected):
        # When Then
        result = abs(test)

        # Expect
        assert result.name == result.unique_name 
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max