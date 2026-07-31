# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from unittest.mock import MagicMock

import numpy as np
import pytest
import scipp as sc
from scipp import UnitError

from easyscience import DescriptorNumber
from easyscience import ObjBase
from easyscience import Parameter
from easyscience import global_object


class TestParameter:
    @pytest.fixture
    def parameter(self) -> Parameter:
        self.mock_callback = MagicMock()
        parameter = Parameter(
            name='name',
            value=1,
            unit='m',
            variance=0.01,
            min=0,
            max=10,
            description='description',
            url='url',
            display_name='display_name',
            callback=self.mock_callback,
            parent=None,
        )
        return parameter

    @pytest.fixture
    def normal_parameter(self) -> Parameter:
        parameter = Parameter(
            name='name',
            value=1,
            unit='m',
            variance=0.01,
            min=0,
            max=10,
        )
        return parameter

    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    def compare_parameters(self, parameter1: Parameter, parameter2: Parameter):
        assert parameter1.value == parameter2.value
        assert parameter1.unit == parameter2.unit
        assert parameter1.variance == parameter2.variance
        assert parameter1.min == parameter2.min
        assert parameter1.max == parameter2.max
        assert parameter1._min.unit == parameter2._min.unit
        assert parameter1._max.unit == parameter2._max.unit

    def test_init(self, parameter: Parameter):
        # When Then Expect
        assert parameter._min.value == 0
        assert parameter._min.unit == 'm'
        assert parameter._max.value == 10
        assert parameter._max.unit == 'm'
        assert parameter._callback == self.mock_callback
        assert parameter._independent == True

        # From super
        assert parameter._scalar.value == 1
        assert parameter._scalar.unit == 'm'
        assert parameter._scalar.variance == 0.01
        assert parameter._name == 'name'
        assert parameter._description == 'description'
        assert parameter._url == 'url'
        assert parameter._display_name == 'display_name'
        assert parameter._fixed == False
        assert parameter._observers == []

    def test_init_value_min_exception(self):
        # When
        mock_callback = MagicMock()
        value = -1

        # Then Expect
        with pytest.raises(ValueError):
            Parameter(
                name='name',
                value=value,
                unit='m',
                variance=0.01,
                min=0,
                max=10,
                description='description',
                url='url',
                display_name='display_name',
                callback=mock_callback,
                parent=None,
            )

    def test_init_value_max_exception(self):
        # When
        mock_callback = MagicMock()
        value = 100

        # Then Expect
        with pytest.raises(ValueError):
            Parameter(
                name='name',
                value=value,
                unit='m',
                variance=0.01,
                min=0,
                max=10,
                description='description',
                url='url',
                display_name='display_name',
                callback=mock_callback,
                parent=None,
            )

    def test_make_dependent_on(self, normal_parameter: Parameter):
        # When
        independent_parameter = Parameter(
            name='independent', value=1, unit='m', variance=0.01, min=0, max=10
        )

        # Then
        normal_parameter.make_dependent_on(
            dependency_expression='2*a', dependency_map={'a': independent_parameter}
        )

        # Expect
        assert normal_parameter._independent == False
        assert normal_parameter.dependency_expression == '2*a'
        assert normal_parameter.dependency_map == {'a': independent_parameter}
        self.compare_parameters(normal_parameter, 2 * independent_parameter)

        # Then
        independent_parameter.value = 2

        # Expect
        normal_parameter.value == 4
        self.compare_parameters(normal_parameter, 2 * independent_parameter)

    def test_dependent_parameter_update_pushes_value_to_callback(self, parameter: Parameter):
        # When # Simulate a calculator bound to the parameter via the callback (issue #281)
        calculator = {'value': parameter._scalar.value}
        self.mock_callback.fget.side_effect = lambda: calculator['value']
        self.mock_callback.fset.side_effect = lambda value: calculator.update(value=value)
        independent_parameter = Parameter(
            name='independent', value=1, unit='m', variance=0.01, min=0, max=10
        )

        # Then
        parameter.make_dependent_on(
            dependency_expression='2*a', dependency_map={'a': independent_parameter}
        )

        # Expect # Making the parameter dependent pushes the computed value to the calculator
        assert calculator['value'] == 2.0
        self.mock_callback.fset.assert_called_with(2.0)
        assert self.mock_callback.fset.call_count == 1

        # Then
        independent_parameter.value = 3

        # Expect # The change of the independent parameter propagates to the calculator
        assert calculator['value'] == 6.0
        self.mock_callback.fset.assert_called_with(6.0)
        assert self.mock_callback.fset.call_count == 2
        # Reading the value must not return a stale calculator value
        assert parameter.value == 6.0

    def test_dependent_parameter_update_pushes_value_to_callback_with_desired_unit(
        self, parameter: Parameter
    ):
        # When # Simulate a calculator bound to the parameter via the callback (issue #281)
        calculator = {'value': parameter._scalar.value}
        self.mock_callback.fget.side_effect = lambda: calculator['value']
        self.mock_callback.fset.side_effect = lambda value: calculator.update(value=value)
        independent_parameter = Parameter(
            name='independent', value=1, unit='m', variance=0.01, min=0, max=10
        )

        # Then
        parameter.make_dependent_on(
            dependency_expression='2*a',
            dependency_map={'a': independent_parameter},
            desired_unit='cm',
        )

        # Expect # The calculator receives the value converted to the desired unit
        assert calculator['value'] == 200.0
        self.mock_callback.fset.assert_called_with(200.0)

        # Then
        independent_parameter.value = 2

        # Expect
        assert calculator['value'] == 400.0
        assert parameter.value == 400.0

    def test_dependent_parameter_make_dependent_on_with_desired_unit(
        self, normal_parameter: Parameter
    ):
        # When
        independent_parameter = Parameter(
            name='independent', value=1, unit='m', variance=0.01, min=0, max=10
        )

        # Then
        normal_parameter.make_dependent_on(
            dependency_expression='2*a',
            dependency_map={'a': independent_parameter},
            desired_unit='cm',
        )

        # Expect
        assert normal_parameter._independent == False
        assert normal_parameter.dependency_expression == '2*a'
        assert normal_parameter.dependency_map == {'a': independent_parameter}

        assert normal_parameter.value == 200 * independent_parameter.value
        assert normal_parameter.unit == 'cm'
        assert (
            normal_parameter.variance == independent_parameter.variance * 4 * 10000
        )  # unit conversion from m to cm squared
        assert normal_parameter.min == 200 * independent_parameter.min
        assert normal_parameter.max == 200 * independent_parameter.max
        assert normal_parameter._min.unit == 'cm'
        assert normal_parameter._max.unit == 'cm'

        # Then
        independent_parameter.value = 2

        # Expect
        assert normal_parameter.value == 200 * independent_parameter.value
        assert normal_parameter.unit == 'cm'
        assert (
            normal_parameter.variance == independent_parameter.variance * 4 * 10000
        )  # unit conversion from m to cm squared
        assert normal_parameter.min == 200 * independent_parameter.min
        assert normal_parameter.max == 200 * independent_parameter.max
        assert normal_parameter._min.unit == 'cm'
        assert normal_parameter._max.unit == 'cm'

        # Then # Change the dependency expression and unit again
        normal_parameter.make_dependent_on(
            dependency_expression='3*a',
            dependency_map={'a': independent_parameter},
            desired_unit='mm',
        )

        # Expect
        assert normal_parameter._independent == False
        assert normal_parameter.dependency_expression == '3*a'
        assert normal_parameter.dependency_map == {'a': independent_parameter}

        assert normal_parameter.value == 3000 * independent_parameter.value
        assert normal_parameter.unit == 'mm'
        assert (
            normal_parameter.variance == independent_parameter.variance * 9 * 1000000
        )  # unit conversion from m to mm squared
        assert normal_parameter.min == 3000 * independent_parameter.min
        assert normal_parameter.max == 3000 * independent_parameter.max
        assert normal_parameter._min.unit == 'mm'
        assert normal_parameter._max.unit == 'mm'

        # Then
        independent_parameter.value = 2

        # Expect
        assert normal_parameter.value == 3000 * independent_parameter.value
        assert normal_parameter.unit == 'mm'
        assert (
            normal_parameter.variance == independent_parameter.variance * 9 * 1000000
        )  # unit conversion from m to mm squared
        assert normal_parameter.min == 3000 * independent_parameter.min
        assert normal_parameter.max == 3000 * independent_parameter.max
        assert normal_parameter._min.unit == 'mm'
        assert normal_parameter._max.unit == 'mm'

    def test_dependent_parameter_make_dependent_on_with_desired_unit_incompatible_unit_raises(
        self, normal_parameter: Parameter
    ):
        # When
        independent_parameter = Parameter(
            name='independent', value=1, unit='m', variance=0.01, min=0, max=10
        )

        # Then Expect
        with pytest.raises(UnitError, match='Failed to convert unit from m to s.'):
            normal_parameter.make_dependent_on(
                dependency_expression='2*a',
                dependency_map={'a': independent_parameter},
                desired_unit='s',
            )

    def test_dependent_parameter_make_dependent_on_with_incorrect_unit_raises(
        self, normal_parameter: Parameter
    ):
        # When
        independent_parameter = Parameter(
            name='independent', value=1, unit='m', variance=0.01, min=0, max=10
        )

        # Then Expect
        with pytest.raises(
            TypeError, match='`desired_unit` must be a string representing a valid unit'
        ):
            normal_parameter.make_dependent_on(
                dependency_expression='2*a',
                dependency_map={'a': independent_parameter},
                desired_unit=123,
            )

    def test_parameter_from_dependency(self, normal_parameter: Parameter):
        # When Then
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
            display_name='display_name',
        )

        # Expect
        assert dependent_parameter._independent == False
        assert dependent_parameter.dependency_expression == '2*a'
        assert dependent_parameter.dependency_map == {'a': normal_parameter}
        assert dependent_parameter.name == 'dependent'
        assert dependent_parameter.display_name == 'display_name'
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        # Then
        normal_parameter.value = 2

        # Expect
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

    def test_parameter_from_dependency_with_desired_unit(self, normal_parameter: Parameter):
        # When Then
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
            display_name='display_name',
            desired_unit='cm',
        )

        # Expect
        assert dependent_parameter._independent == False
        assert dependent_parameter.dependency_expression == '2*a'
        assert dependent_parameter.dependency_map == {'a': normal_parameter}
        assert dependent_parameter.name == 'dependent'
        assert dependent_parameter.display_name == 'display_name'

        assert dependent_parameter.value == 200 * normal_parameter.value
        assert dependent_parameter.unit == 'cm'
        assert (
            dependent_parameter.variance == normal_parameter.variance * 4 * 10000
        )  # unit conversion from m to cm squared
        assert dependent_parameter.min == 200 * normal_parameter.min
        assert dependent_parameter.max == 200 * normal_parameter.max
        assert dependent_parameter._min.unit == 'cm'
        assert dependent_parameter._max.unit == 'cm'

        # Then
        normal_parameter.value = 2

        # Expect
        assert dependent_parameter.value == 200 * normal_parameter.value
        assert dependent_parameter.unit == 'cm'
        assert (
            dependent_parameter.variance == normal_parameter.variance * 4 * 10000
        )  # unit conversion from m to cm squared
        assert dependent_parameter.min == 200 * normal_parameter.min
        assert dependent_parameter.max == 200 * normal_parameter.max
        assert dependent_parameter._min.unit == 'cm'
        assert dependent_parameter._max.unit == 'cm'

    def test_parameter_from_dependency_with_desired_unit_incompatible_unit_raises(
        self, normal_parameter: Parameter
    ):
        # When Then Expect
        with pytest.raises(UnitError):
            dependent_parameter = Parameter.from_dependency(
                name='dependent',
                dependency_expression='2*a',
                dependency_map={'a': normal_parameter},
                display_name='display_name',
                desired_unit='s',
            )

    def test_dependent_parameter_with_unique_name(self, clear, normal_parameter: Parameter):
        # When Then
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*"Parameter_0"',
        )

        # Expect
        assert dependent_parameter.dependency_expression == '2*"Parameter_0"'
        assert dependent_parameter.dependency_map == {'__Parameter_0__': normal_parameter}
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        # Then
        normal_parameter.value = 2

        # Expect
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

    def test_process_dependency_unique_names_double_quotes(
        self, clear, normal_parameter: Parameter
    ):
        # When
        independent_parameter = Parameter(
            name='independent',
            value=1,
            unit='m',
            variance=0.01,
            min=0,
            max=10,
            unique_name='Special_name',
        )
        normal_parameter._dependency_map = {}

        # Then
        normal_parameter._process_dependency_unique_names(dependency_expression='2*"Special_name"')

        # Expect
        assert normal_parameter._dependency_map == {'__Special_name__': independent_parameter}
        assert normal_parameter._clean_dependency_string == '2*__Special_name__'

    def test_process_dependency_unique_names_single_quotes(
        self, clear, normal_parameter: Parameter
    ):
        # When
        independent_parameter = Parameter(
            name='independent',
            value=1,
            unit='m',
            variance=0.01,
            min=0,
            max=10,
            unique_name='Special_name',
        )
        independent_parameter_2 = Parameter(
            name='independent_2',
            value=1,
            unit='m',
            variance=0.01,
            min=0,
            max=10,
            unique_name='Special_name_2',
        )
        normal_parameter._dependency_map = {}

        # Then
        normal_parameter._process_dependency_unique_names(
            dependency_expression="'Special_name' + 'Special_name_2'"
        )

        # Expect
        assert normal_parameter._dependency_map == {
            '__Special_name__': independent_parameter,
            '__Special_name_2__': independent_parameter_2,
        }
        assert normal_parameter._clean_dependency_string == '__Special_name__ + __Special_name_2__'

    def test_process_dependency_unique_names_exception_unique_name_does_not_exist(
        self, clear, normal_parameter: Parameter
    ):
        # When
        normal_parameter._dependency_map = {}

        # Then Expect
        with pytest.raises(
            ValueError,
            match='A Parameter with unique_name Special_name does not exist. Please check your dependency expression.',
        ):
            normal_parameter._process_dependency_unique_names(
                dependency_expression='2*"Special_name"'
            )

    def test_process_dependency_unique_names_exception_not_a_descriptorNumber(
        self, clear, normal_parameter: Parameter
    ):
        # When
        normal_parameter._dependency_map = {}
        base_obj = ObjBase(name='ObjBase', unique_name='base_obj')

        # Then Expect
        with pytest.raises(
            ValueError,
            match='The object with unique_name base_obj is not a Parameter or DescriptorNumber. Please check your dependency expression.',
        ):
            normal_parameter._process_dependency_unique_names(dependency_expression='2*"base_obj"')

    @pytest.mark.parametrize(
        'dependency_expression, dependency_map',
        [
            (2, {'a': Parameter(name='a', value=1)}),
            ('2*a', ['a', Parameter(name='a', value=1)]),
            ('2*a', {4: Parameter(name='a', value=1)}),
            ('2*a', {'a': ObjBase(name='a')}),
        ],
        ids=[
            'dependency_expression_not_a_string',
            'dependency_map_not_a_dict',
            'dependency_map_keys_not_strings',
            'dependency_map_values_not_descriptor_number',
        ],
    )
    def test_parameter_from_dependency_input_exceptions(
        self, dependency_expression, dependency_map
    ):
        # When Then Expect
        with pytest.raises(TypeError):
            Parameter.from_dependency(
                name='dependent',
                dependency_expression=dependency_expression,
                dependency_map=dependency_map,
            )

    @pytest.mark.parametrize(
        'dependency_expression, error',
        [
            ('2*a + b', NameError),
            ('2*a + 3*', SyntaxError),
            ('2 + 2', TypeError),
            ('2*"special_name"', ValueError),
        ],
        ids=[
            'parameter_not_in_map',
            'invalid_dependency_expression',
            'result_not_a_descriptor_number',
            'unique_name_does_not_exist',
        ],
    )
    def test_parameter_make_dependent_on_exceptions_cleanup_previously_dependent(
        self, normal_parameter, dependency_expression, error
    ):
        # When
        independent_parameter = Parameter(name='independent', value=10, unit='s', variance=0.02)
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='best',
            dependency_map={'best': independent_parameter},
        )
        # Then Expect
        # Check that the correct error is raised
        with pytest.raises(error):
            dependent_parameter.make_dependent_on(
                dependency_expression=dependency_expression,
                dependency_map={'a': normal_parameter},
            )
        # Check that everything is properly cleaned up
        assert normal_parameter._observers == []
        assert dependent_parameter.independent == False
        assert dependent_parameter.dependency_expression == 'best'
        assert dependent_parameter.dependency_map == {'best': independent_parameter}
        independent_parameter.value = 50
        self.compare_parameters(dependent_parameter, independent_parameter)

    def test_parameter_make_dependent_on_exceptions_cleanup_previously_independent(
        self, normal_parameter
    ):
        # When
        independent_parameter = Parameter(name='independent', value=10, unit='s', variance=0.02)
        # Then Expect
        # Check that the correct error is raised
        with pytest.raises(NameError):
            independent_parameter.make_dependent_on(
                dependency_expression='2*a + b',
                dependency_map={'a': normal_parameter},
            )
        # Check that everything is properly cleaned up
        assert normal_parameter._observers == []
        assert independent_parameter.independent == True
        normal_parameter.value = 50
        assert independent_parameter.value == 10

    def test_dependent_parameter_updates(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        normal_parameter.value = 2
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        normal_parameter.variance = 0.02
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        normal_parameter.error = 0.2
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        normal_parameter.convert_unit('cm')
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        normal_parameter.min = 1
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        normal_parameter.max = 300
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

    def test_dependent_parameter_indirect_updates(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )
        dependent_parameter_2 = Parameter.from_dependency(
            name='dependent_2',
            dependency_expression='10*a',
            dependency_map={'a': normal_parameter},
        )
        dependent_parameter_3 = Parameter.from_dependency(
            name='dependent_3',
            dependency_expression='b+c',
            dependency_map={'b': dependent_parameter, 'c': dependent_parameter_2},
        )
        # Then
        normal_parameter.value = 2

        # Expect
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)
        self.compare_parameters(dependent_parameter_2, 10 * normal_parameter)
        self.compare_parameters(
            dependent_parameter_3, 2 * normal_parameter + 10 * normal_parameter
        )

    def test_dependent_parameter_cyclic_dependencies(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )
        dependent_parameter_2 = Parameter.from_dependency(
            name='dependent_2',
            dependency_expression='2*b',
            dependency_map={'b': dependent_parameter},
        )

        # Then Expect
        with pytest.raises(RuntimeError):
            normal_parameter.make_dependent_on(
                dependency_expression='2*c', dependency_map={'c': dependent_parameter_2}
            )
        # Check that everything is properly cleaned up
        assert dependent_parameter_2._observers == []
        assert normal_parameter.independent == True
        assert normal_parameter.value == 1
        normal_parameter.value = 50
        self.compare_parameters(dependent_parameter_2, 4 * normal_parameter)

    def test_dependent_parameter_logical_dependency(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='a if a.value > 0 else -a',
            dependency_map={'a': normal_parameter},
        )
        self.compare_parameters(dependent_parameter, normal_parameter)

        # Then
        normal_parameter.value = -2

        # Expect
        self.compare_parameters(dependent_parameter, -normal_parameter)

    def test_dependent_parameter_return_is_descriptor_number(self):
        # When
        descriptor_number = DescriptorNumber(name='descriptor', value=1, unit='m', variance=0.01)

        # Then
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*descriptor',
            dependency_map={'descriptor': descriptor_number},
        )

        # Expect
        assert dependent_parameter.value == 2 * descriptor_number.value
        assert dependent_parameter.unit == descriptor_number.unit
        assert dependent_parameter.variance == 0.04
        assert dependent_parameter.min == 2 * descriptor_number.value
        assert dependent_parameter.max == 2 * descriptor_number.value

    def test_dependent_parameter_division_expression_order(self):
        """Test that division expressions work regardless of operand order.

        This is a regression test for https://github.com/easyscience/corelib/issues/190
        where 'a / b' would fail with an observer notification error while '1/b * a'
        would work correctly.
        """
        # When
        angstrom = DescriptorNumber('angstrom', 1e-10, unit='m')
        jump_length = Parameter(
            name='jump_length',
            value=float(1.0),
            fixed=False,
            unit='angstrom',
        )

        expression = 'jump_length / angstrom'
        dependency_map = {
            'jump_length': jump_length,
            'angstrom': angstrom,
        }

        # Calculate the expected result from direct division
        expected_result = jump_length / angstrom
        expected_value = expected_result.value

        # Then - This should not raise an error
        dependent_param = Parameter(name='a', value=1.0)
        dependent_param.make_dependent_on(
            dependency_expression=expression,
            dependency_map=dependency_map,
        )

        # Expect - The dependent parameter should have the same value as the direct division
        assert dependent_param.value == pytest.approx(expected_value)

        # Also test the alternative expression that previously worked
        expression_alt = '1/angstrom * jump_length'
        dependent_param_alt = Parameter(name='b', value=1.0)
        dependent_param_alt.make_dependent_on(
            dependency_expression=expression_alt,
            dependency_map=dependency_map,
        )
        assert dependent_param_alt.value == pytest.approx(expected_value)

    def test_dependent_parameter_overwrite_dependency(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        # Then
        normal_parameter_2 = Parameter(
            name='a2', value=-2, unit='m', variance=0.01, min=-10, max=0
        )
        dependent_parameter.make_dependent_on(
            dependency_expression='3*a2', dependency_map={'a2': normal_parameter_2}
        )
        normal_parameter.value = 3

        # Expect
        self.compare_parameters(dependent_parameter, 3 * normal_parameter_2)
        assert dependent_parameter.dependency_expression == '3*a2'
        assert dependent_parameter.dependency_map == {'a2': normal_parameter_2}
        assert normal_parameter._observers == []

    def test_make_independent(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )
        assert dependent_parameter.independent == False
        self.compare_parameters(dependent_parameter, 2 * normal_parameter)

        # Then
        dependent_parameter.make_independent()
        normal_parameter.value = 5

        # Expect
        assert dependent_parameter.independent == True
        assert normal_parameter._observers == []
        assert dependent_parameter.value == 2

    def test_make_independent_exception(self, normal_parameter: Parameter):
        # When Then Expect
        with pytest.raises(AttributeError):
            normal_parameter.make_independent()

    def test_independent_setter(self, normal_parameter: Parameter):
        # When Then Expect
        with pytest.raises(AttributeError):
            normal_parameter.independent = False

    def test_independent_parameter_dependency_expression(self, normal_parameter: Parameter):
        # When Then Expect
        with pytest.raises(AttributeError):
            normal_parameter.dependency_expression

    def test_dependent_parameter_dependency_expression_setter(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent_parameter.dependency_expression = '3*a'

    def test_independent_parameter_dependency_map(self, normal_parameter: Parameter):
        # When Then Expect
        with pytest.raises(AttributeError):
            normal_parameter.dependency_map

    def test_dependent_parameter_dependency_map_setter(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent_parameter.dependency_map = {'a': normal_parameter}

    def test_min(self, parameter: Parameter):
        # When Then Expect
        assert parameter.min == 0

    def test_set_min(self, parameter: Parameter):
        # When Then
        self.mock_callback.fget.return_value = 1.0  # Ensure fget returns a scalar value

        parameter.min = 0.1

        # Expect
        assert parameter.min == 0.1

    def test_set_min_dependent_parameter(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent_parameter.min = 0.1

    def test_set_min_exception(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.min = 10

    def test_set_max(self, parameter: Parameter):
        # When Then
        parameter.max = 10

        # Expect
        assert parameter.max == 10

    def test_set_max_dependent_parameter(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent_parameter.max = 10

    def test_set_max_exception(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(ValueError):
            parameter.max = 0.1

    def test_convert_unit(self, parameter: Parameter):
        # When Then
        parameter.convert_unit('mm')

        # Expect
        assert parameter._min.value == 0
        assert parameter._min.unit == 'mm'
        assert parameter._max.value == 10000
        assert parameter._max.unit == 'mm'

    def test_set_desired_unit(self, normal_parameter: Parameter):
        # When Then
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
            display_name='display_name',
        )

        # Then
        dependent_parameter.set_desired_unit('cm')

        # Expect

        assert dependent_parameter.value == 200 * normal_parameter.value
        assert dependent_parameter.unit == 'cm'
        assert (
            dependent_parameter.variance == normal_parameter.variance * 4 * 10000
        )  # unit conversion from m to cm squared
        assert dependent_parameter.min == 200 * normal_parameter.min
        assert dependent_parameter.max == 200 * normal_parameter.max
        assert dependent_parameter._min.unit == 'cm'
        assert dependent_parameter._max.unit == 'cm'

        # Then
        normal_parameter.value = 2

        # Expect
        assert dependent_parameter.value == 200 * normal_parameter.value
        assert dependent_parameter.unit == 'cm'
        assert (
            dependent_parameter.variance == normal_parameter.variance * 4 * 10000
        )  # unit conversion from m to cm squared
        assert dependent_parameter.min == 200 * normal_parameter.min
        assert dependent_parameter.max == 200 * normal_parameter.max
        assert dependent_parameter._min.unit == 'cm'
        assert dependent_parameter._max.unit == 'cm'

    def test_set_desired_unit_incompatible_units_raises(self, normal_parameter: Parameter):
        # When Then
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
            display_name='display_name',
        )

        # Then Expect
        with pytest.raises(
            UnitError,
            match='Failed to convert unit from m to s.',
        ):
            dependent_parameter.set_desired_unit('s')

    def test_set_desired_unit_None(self, normal_parameter: Parameter):
        # When Then
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
            display_name='display_name',
            desired_unit='cm',
        )

        # EXPECT
        assert dependent_parameter.unit == 'cm'

        # Then
        dependent_parameter.set_desired_unit(None)

        # EXPECT
        assert dependent_parameter.value == 2 * normal_parameter.value
        assert dependent_parameter.unit == 'm'

    def test_set_desired_unit_independent_parameter_raises(self, normal_parameter: Parameter):
        # When Then Expect
        with pytest.raises(
            AttributeError,
            match='This is an independent parameter, desired unit can only be set for dependent parameters.',
        ):
            normal_parameter.set_desired_unit('cm')

    def test_set_desired_unit_incorrect_unit_type_raises(self, normal_parameter: Parameter):
        # When Then
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
            display_name='display_name',
        )

        # Then Expect
        with pytest.raises(TypeError, match='must be a string'):
            dependent_parameter.set_desired_unit(5)

    def test_set_fixed(self, parameter: Parameter):
        # When Then
        parameter.fixed = True

        # Expect
        assert parameter.fixed == True

    def test_set_fixed_dependent_parameter(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent_parameter.fixed = True

    @pytest.mark.parametrize('fixed', ['True', 1])
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
        assert (
            repr(parameter) == "<Parameter 'name': 1.0000 ± 0.1000 m (fixed), bounds=[0.0:10.0]>"
        )

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

    def test_set_value_dependent_parameter(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent_parameter.value = 3

    def test_set_full_value(self, parameter: Parameter):
        # When Then Expect
        with pytest.raises(AttributeError):
            parameter.full_value = sc.scalar(2, unit='s')

    def test_set_variance_dependent_parameter(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent_parameter.variance = 0.1

    def test_set_error_dependent_parameter(self, normal_parameter: Parameter):
        # When
        dependent_parameter = Parameter.from_dependency(
            name='dependent',
            dependency_expression='2*a',
            dependency_map={'a': normal_parameter},
        )

        # Then Expect
        with pytest.raises(AttributeError):
            dependent_parameter.error = 0.1

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
        assert parameter_copy._independent == parameter._independent

    @pytest.mark.parametrize(
        'test, expected, expected_reverse',
        [
            (
                Parameter('test', 2, 'm', 0.01, -10, 20),
                Parameter('name + test', 3, 'm', 0.02, -10, 30),
                Parameter('test + name', 3, 'm', 0.02, -10, 30),
            ),
            (
                Parameter('test', 2, 'm', 0.01),
                Parameter('name + test', 3, 'm', 0.02, min=-np.inf, max=np.inf),
                Parameter('test + name', 3, 'm', 0.02, min=-np.inf, max=np.inf),
            ),
            (
                Parameter('test', 2, 'cm', 0.01, -10, 10),
                Parameter('name + test', 1.02, 'm', 0.010001, -0.1, 10.1),
                Parameter('test + name', 102, 'cm', 100.01, -10, 1010),
            ),
        ],
        ids=['regular', 'no_bounds', 'unit_conversion'],
    )
    def test_addition_with_parameter(
        self,
        parameter: Parameter,
        test: Parameter,
        expected: Parameter,
        expected_reverse: Parameter,
    ):
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

        assert parameter.unit == 'm'

    def test_addition_with_scalar(self):
        # When
        parameter = Parameter(name='name', value=1, variance=0.01, min=0, max=10)

        # Then
        result = parameter + 1.0
        result_reverse = 1.0 + parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == 2.0
        assert result.unit == 'dimensionless'
        assert result.variance == 0.01
        assert result.min == 1.0
        assert result.max == 11.0

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == 2.0
        assert result_reverse.unit == 'dimensionless'
        assert result_reverse.variance == 0.01
        assert result_reverse.min == 1.0
        assert result_reverse.max == 11.0

    def test_addition_with_descriptor_number(self, parameter: Parameter):
        # When
        parameter._callback = property()
        descriptor_number = DescriptorNumber(name='test', value=1, variance=0.1, unit='cm')

        # Then
        result = parameter + descriptor_number
        result_reverse = descriptor_number + parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == result.unique_name
        assert result.value == 1.01
        assert result.unit == 'm'
        assert result.variance == 0.01001
        assert result.min == 0.01
        assert result.max == 10.01

        assert type(result_reverse) == Parameter
        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == 101.0
        assert result_reverse.unit == 'cm'
        assert result_reverse.variance == 100.1
        assert result_reverse.min == 1
        assert result_reverse.max == 1001

        assert parameter.unit == 'm'
        assert descriptor_number.unit == 'cm'

    @pytest.mark.parametrize(
        'test',
        [
            1.0,
            Parameter(
                'test',
                2,
                's',
            ),
        ],
        ids=['add_scalar_to_unit', 'incompatible_units'],
    )
    def test_addition_exception(self, parameter: Parameter, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = parameter + test
        with pytest.raises(UnitError):
            result_reverse = test + parameter

    @pytest.mark.parametrize(
        'test, expected, expected_reverse',
        [
            (
                Parameter('test', 2, 'm', 0.01, -20, 20),
                Parameter('name - test', -1, 'm', 0.02, -20, 30),
                Parameter('test - name', 1, 'm', 0.02, -30, 20),
            ),
            (
                Parameter('test', 2, 'm', 0.01),
                Parameter('name - test', -1, 'm', 0.02, min=-np.inf, max=np.inf),
                Parameter('test - name', 1, 'm', 0.02, min=-np.inf, max=np.inf),
            ),
            (
                Parameter('test', 2, 'cm', 0.01, -10, 10),
                Parameter('name - test', 0.98, 'm', 0.010001, -0.1, 10.1),
                Parameter('test - name', -98, 'cm', 100.01, -1010, 10),
            ),
        ],
        ids=['regular', 'no_bounds', 'unit_conversion'],
    )
    def test_subtraction_with_parameter(
        self,
        parameter: Parameter,
        test: Parameter,
        expected: Parameter,
        expected_reverse: Parameter,
    ):
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

        assert parameter.unit == 'm'

    def test_subtraction_with_parameter_nan_cases(self):
        # When
        parameter = Parameter(name='name', value=1, variance=0.01, min=-np.inf, max=np.inf)
        test = Parameter(name='test', value=2, variance=0.01, min=-np.inf, max=np.inf)

        # Then
        result = parameter - test
        result_reverse = test - parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == -1.0
        assert result.unit == 'dimensionless'
        assert result.variance == 0.02
        assert result.min == -np.inf
        assert result.max == np.inf

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == 1.0
        assert result_reverse.unit == 'dimensionless'
        assert result_reverse.variance == 0.02
        assert result_reverse.min == -np.inf
        assert result_reverse.max == np.inf

    def test_subtraction_with_scalar(self):
        # When
        parameter = Parameter(name='name', value=2, variance=0.01, min=0, max=10)

        # Then
        result = parameter - 1.0
        result_reverse = 1.0 - parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == 1.0
        assert result.unit == 'dimensionless'
        assert result.variance == 0.01
        assert result.min == -1.0
        assert result.max == 9.0

        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == -1.0
        assert result_reverse.unit == 'dimensionless'
        assert result_reverse.variance == 0.01
        assert result_reverse.min == -9.0
        assert result_reverse.max == 1.0

    def test_subtraction_with_descriptor_number(self, parameter: Parameter):
        # When
        parameter._callback = property()
        descriptor_number = DescriptorNumber(name='test', value=1, variance=0.1, unit='cm')

        # Then
        result = parameter - descriptor_number
        result_reverse = descriptor_number - parameter

        # Expect
        assert type(result) == Parameter
        assert result.name == result.unique_name
        assert result.value == 0.99
        assert result.unit == 'm'
        assert result.variance == 0.01001
        assert result.min == -0.01
        assert result.max == 9.99

        assert type(result_reverse) == Parameter
        assert result_reverse.name == result_reverse.unique_name
        assert result_reverse.value == -99.0
        assert result_reverse.unit == 'cm'
        assert result_reverse.variance == 100.1
        assert result_reverse.min == -999
        assert result_reverse.max == 1

        assert parameter.unit == 'm'
        assert descriptor_number.unit == 'cm'

    @pytest.mark.parametrize(
        'test',
        [
            1.0,
            Parameter(
                'test',
                2,
                's',
            ),
        ],
        ids=['sub_scalar_to_unit', 'incompatible_units'],
    )
    def test_subtraction_exception(self, parameter: Parameter, test):
        # When Then Expect
        with pytest.raises(UnitError):
            result = parameter - test
        with pytest.raises(UnitError):
            result_reverse = test - parameter

    @pytest.mark.parametrize(
        'test, expected, expected_reverse',
        [
            (
                Parameter('test', 2, 'm', 0.01, -10, 20),
                Parameter('name * test', 2, 'm^2', 0.05, -100, 200),
                Parameter('test * name', 2, 'm^2', 0.05, -100, 200),
            ),
            (
                Parameter('test', 2, 'm', 0.01),
                Parameter('name * test', 2, 'm^2', 0.05, min=-np.inf, max=np.inf),
                Parameter('test * name', 2, 'm^2', 0.05, min=-np.inf, max=np.inf),
            ),
            (
                Parameter('test', 2, 'dm', 0.01, -10, 20),
                Parameter('name * test', 0.2, 'm^2', 0.0005, -10, 20),
                Parameter('test * name', 0.2, 'm^2', 0.0005, -10, 20),
            ),
        ],
        ids=['regular', 'no_bounds', 'base_unit_conversion'],
    )
    def test_multiplication_with_parameter(
        self,
        parameter: Parameter,
        test: Parameter,
        expected: Parameter,
        expected_reverse: Parameter,
    ):
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

    @pytest.mark.parametrize(
        'test, expected, expected_reverse',
        [
            (
                Parameter('test', 0, '', 0.01, -10, 0),
                Parameter('name * test', 0.0, 'dimensionless', 0.01, -np.inf, 0),
                Parameter('test * name', 0, 'dimensionless', 0.01, -np.inf, 0),
            ),
            (
                Parameter('test', 0, '', 0.01, 0, 10),
                Parameter('name * test', 0.0, 'dimensionless', 0.01, 0, np.inf),
                Parameter('test * name', 0, 'dimensionless', 0.01, 0, np.inf),
            ),
        ],
        ids=['zero_min', 'zero_max'],
    )
    def test_multiplication_with_parameter_nan_cases(self, test, expected, expected_reverse):
        # When
        parameter = Parameter(name='name', value=1, variance=0.01, min=1, max=np.inf)

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

    @pytest.mark.parametrize(
        'test, expected, expected_reverse',
        [
            (
                DescriptorNumber(name='test', value=2, variance=0.1, unit='cm'),
                Parameter('name * test', 2, 'dm^2', 0.14, 0, 20),
                Parameter('test * name', 2, 'dm^2', 0.14, 0, 20),
            ),
            (
                DescriptorNumber(name='test', value=0, variance=0.1, unit='cm'),
                DescriptorNumber('name * test', 0, 'dm^2', 0.1),
                DescriptorNumber('test * name', 0, 'dm^2', 0.1),
            ),
        ],
        ids=['regular', 'zero_value'],
    )
    def test_multiplication_with_descriptor_number(
        self, parameter: Parameter, test, expected, expected_reverse
    ):
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

    @pytest.mark.parametrize(
        'test, expected, expected_reverse',
        [
            (
                2,
                Parameter('name * 2', 2, 'm', 0.04, 0, 20),
                Parameter('2 * name', 2, 'm', 0.04, 0, 20),
            ),
            (
                0,
                DescriptorNumber('name * 0', 0, 'm', 0),
                DescriptorNumber('0 * name', 0, 'm', 0),
            ),
        ],
        ids=['regular', 'zero_value'],
    )
    def test_multiplication_with_scalar(
        self, parameter: Parameter, test, expected, expected_reverse
    ):
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

    @pytest.mark.parametrize(
        'test, expected, expected_reverse',
        [
            (
                Parameter('test', 2, 's', 0.01, -10, 20),
                Parameter('name / test', 0.5, 'm/s', 0.003125, -np.inf, np.inf),
                Parameter('test / name', 2, 's/m', 0.05, -np.inf, np.inf),
            ),
            (
                Parameter('test', 2, 's', 0.01, 0, 20),
                Parameter('name / test', 0.5, 'm/s', 0.003125, 0.0, np.inf),
                Parameter('test / name', 2, 's/m', 0.05, 0.0, np.inf),
            ),
            (
                Parameter('test', -2, 's', 0.01, -10, 0),
                Parameter('name / test', -0.5, 'm/s', 0.003125, -np.inf, 0.0),
                Parameter('test / name', -2, 's/m', 0.05, -np.inf, 0.0),
            ),
        ],
        ids=['crossing_zero', 'only_positive', 'only_negative'],
    )
    def test_division_with_parameter(self, parameter: Parameter, test, expected, expected_reverse):
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

    @pytest.mark.parametrize(
        'first, second, expected',
        [
            (
                Parameter('name', 1, 'm', 0.01, -10, 20),
                Parameter('test', -2, 's', 0.01, -10, 0),
                Parameter('name / test', -0.5, 'm/s', 0.003125, -np.inf, np.inf),
            ),
            (
                Parameter('name', -10, 'm', 0.01, -20, -10),
                Parameter('test', -2, 's', 0.01, -10, 0),
                Parameter('name / test', 5.0, 'm/s', 0.065, 1, np.inf),
            ),
            (
                Parameter('name', 10, 'm', 0.01, 10, 20),
                Parameter('test', -20, 's', 0.01, -20, -10),
                Parameter('name / test', -0.5, 'm/s', 3.125e-5, -2, -0.5),
            ),
        ],
        ids=[
            'first_crossing_zero_second_negative_0',
            'both_negative_second_negative_0',
            'finite_limits',
        ],
    )
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

    @pytest.mark.parametrize(
        'test, expected, expected_reverse',
        [
            (
                DescriptorNumber(name='test', value=2, variance=0.1, unit='s'),
                Parameter('name / test', 0.5, 'm/s', 0.00875, 0, 5),
                Parameter('test / name', 2, 's/m', 0.14, 0.2, np.inf),
            ),
            (
                2,
                Parameter('name / 2', 0.5, 'm', 0.0025, 0, 5),
                Parameter('2 / name', 2, 'm**-1', 0.04, 0.2, np.inf),
            ),
        ],
        ids=['descriptor_number', 'number'],
    )
    def test_division_with_descriptor_number_and_number(
        self, parameter: Parameter, test, expected, expected_reverse
    ):
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

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                DescriptorNumber(name='test', value=0, variance=0.1, unit='s'),
                DescriptorNumber('test / name', 0.0, 's/m', 0.1),
            ),
            (0, DescriptorNumber('0 / name', 0.0, '1/m', 0.0)),
        ],
        ids=['descriptor_number', 'number'],
    )
    def test_zero_value_divided_by_parameter(self, parameter: Parameter, test, expected):
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

    @pytest.mark.parametrize(
        'first, second, expected',
        [
            (
                DescriptorNumber('name', 1, 'm', 0.01),
                Parameter('test', 2, 's', 0.1, -10, 10),
                Parameter('name / test', 0.5, 'm/s', 0.00875, -np.inf, np.inf),
            ),
            (
                DescriptorNumber('name', -1, 'm', 0.01),
                Parameter('test', 2, 's', 0.1, 0, 10),
                Parameter('name / test', -0.5, 'm/s', 0.00875, -np.inf, -0.1),
            ),
            (
                DescriptorNumber('name', 1, 'm', 0.01),
                Parameter('test', -2, 's', 0.1, -10, 0),
                Parameter('name / test', -0.5, 'm/s', 0.00875, -np.inf, -0.1),
            ),
            (
                DescriptorNumber('name', -1, 'm', 0.01),
                Parameter('test', -2, 's', 0.1, -10, 0),
                Parameter('name / test', 0.5, 'm/s', 0.00875, 0.1, np.inf),
            ),
            (
                DescriptorNumber('name', 1, 'm', 0.01),
                Parameter('test', 2, 's', 0.1, 1, 10),
                Parameter('name / test', 0.5, 'm/s', 0.00875, 0.1, 1),
            ),
        ],
        ids=[
            'crossing_zero',
            'positive_0_with_negative',
            'negative_0_with_positive',
            'negative_0_with_negative',
            'finite_limits',
        ],
    )
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

    @pytest.mark.parametrize(
        'test',
        [0, DescriptorNumber('test', 0, 's', 0.1)],
        ids=['number', 'descriptor_number'],
    )
    def test_divide_parameter_by_zero(self, parameter: Parameter, test):
        # When
        parameter._callback = property()

        # Then Expect
        with pytest.raises(ZeroDivisionError):
            result = parameter / test

    def test_divide_by_zero_value_parameter(self):
        # When
        descriptor = DescriptorNumber('test', 1, 's', 0.1)
        parameter = Parameter('name', 0, 'm', 0.01)

        # Then Expect
        with pytest.raises(ZeroDivisionError):
            result = descriptor / parameter

    @pytest.mark.parametrize(
        'test, expected',
        [
            (3, Parameter('name ** 3', 125, 'm^3', 281.25, -125, 1000)),
            (2, Parameter('name ** 2', 25, 'm^2', 5.0, 0, 100)),
            (-1, Parameter('name ** -1', 0.2, '1/m', 8e-5, -np.inf, np.inf)),
            (-2, Parameter('name ** -2', 0.04, '1/m^2', 1.28e-5, 0, np.inf)),
            (0, DescriptorNumber('name ** 0', 1, 'dimensionless', 0)),
            (
                DescriptorNumber('test', 2),
                Parameter('name ** test', 25, 'm^2', 5.0, 0, 100),
            ),
        ],
        ids=[
            'power_3',
            'power_2',
            'power_-1',
            'power_-2',
            'power_0',
            'power_descriptor_number',
        ],
    )
    def test_power_of_parameter(self, test, expected):
        # When
        parameter = Parameter('name', 5, 'm', 0.05, -5, 10)

        # Then
        result = parameter**test

        # Expect
        assert type(result) == type(expected)
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        if isinstance(result, Parameter):
            assert result.min == expected.min
            assert result.max == expected.max

    @pytest.mark.parametrize(
        'test, exponent, expected',
        [
            (
                Parameter('name', 5, 'm', 0.05, 0, 10),
                -1,
                Parameter('name ** -1', 0.2, '1/m', 8e-5, 0.1, np.inf),
            ),
            (
                Parameter('name', -5, 'm', 0.05, -5, 0),
                -1,
                Parameter('name ** -1', -0.2, '1/m', 8e-5, -np.inf, -0.2),
            ),
            (
                Parameter('name', 5, 'm', 0.05, 5, 10),
                -1,
                Parameter('name ** -1', 0.2, '1/m', 8e-5, 0.1, 0.2),
            ),
            (
                Parameter('name', -5, 'm', 0.05, -10, -5),
                -1,
                Parameter('name ** -1', -0.2, '1/m', 8e-5, -0.2, -0.1),
            ),
            (
                Parameter('name', -5, 'm', 0.05, -10, -5),
                -2,
                Parameter('name ** -2', 0.04, '1/m^2', 1.28e-5, 0.01, 0.04),
            ),
            (
                Parameter('name', 5, '', 0.1, 1, 10),
                0.3,
                Parameter(
                    'name ** 0.3',
                    1.6206565966927624,
                    '',
                    0.0009455500095853564,
                    1,
                    1.9952623149688795,
                ),
            ),
            (
                Parameter('name', 5, '', 0.1),
                0.5,
                Parameter('name ** 0.5', 2.23606797749979, '', 0.005, 0, np.inf),
            ),
        ],
        ids=[
            '0_positive',
            'negative_0',
            'both_positive',
            'both_negative_invert',
            'both_negative_invert_square',
            'fractional',
            'fractional_negative_limit',
        ],
    )
    def test_power_of_diffent_parameters(self, test, exponent, expected):
        # When Then
        result = test**exponent

        # Expect
        assert result.name == result.unique_name
        assert result.value == expected.value
        assert result.unit == expected.unit
        assert result.variance == expected.variance
        assert result.min == expected.min
        assert result.max == expected.max

    @pytest.mark.parametrize(
        'parameter, exponent, expected',
        [
            (
                Parameter('name', 5, 'm'),
                DescriptorNumber('test', 2, unit='s'),
                UnitError,
            ),
            (
                Parameter('name', 5, 'm'),
                DescriptorNumber('test', 2, variance=0.01),
                ValueError,
            ),
            (Parameter('name', 5, 'm'), 0.5, UnitError),
            (Parameter('name', -5, ''), 0.5, ValueError),
        ],
        ids=[
            'exponent_unit',
            'exponent_variance',
            'exponent_fractional',
            'negative_base_fractional',
        ],
    )
    def test_power_exceptions(self, parameter, exponent, expected):
        # When Then Expect
        with pytest.raises(expected):
            result = parameter**exponent

    def test_negation(self):
        # When
        parameter = Parameter('name', 5, 'm', 0.05, -5, 10)

        # Then
        result = -parameter

        # Expect
        assert result.name == result.unique_name
        assert result.value == -5
        assert result.unit == 'm'
        assert result.variance == 0.05
        assert result.min == -10
        assert result.max == 5

    @pytest.mark.parametrize(
        'test, expected',
        [
            (
                Parameter('name', -5, 'm', 0.05, -10, -5),
                Parameter('abs(name)', 5, 'm', 0.05, 5, 10),
            ),
            (
                Parameter('name', 5, 'm', 0.05, -10, 10),
                Parameter('abs(name)', 5, 'm', 0.05, 0, 10),
            ),
        ],
        ids=['pure_negative', 'crossing_zero'],
    )
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
