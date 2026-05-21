# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging
from unittest.mock import MagicMock

import pytest

from easyscience import global_object
from easyscience.fitting.calculators.interface_factory import InterfaceFactoryTemplate
from easyscience.fitting.calculators.interface_factory import ItemContainer


class TestInterfaceFactoryTemplate:
    @pytest.fixture
    def clear(self):
        """Clear global map to avoid test contamination"""
        global_object.map._clear()
        yield
        global_object.map._clear()

    @pytest.fixture
    def mock_interface1(self):
        """Create a mock interface class"""

        class MockInterface1:
            name = 'MockInterface1'

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def fit_func(self, *args, **kwargs):
                return 'result1'

            def create(self, model):
                return []

        return MockInterface1

    @pytest.fixture
    def mock_interface2(self):
        """Create a second mock interface class"""

        class MockInterface2:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def fit_func(self, *args, **kwargs):
                return 'result2'

            def create(self, model):
                return []

        return MockInterface2

    @pytest.fixture
    def mock_interface_with_custom_name(self):
        """Create a mock interface with custom name attribute"""

        class MockInterfaceCustom:
            name = 'CustomName'

            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def fit_func(self, *args, **kwargs):
                return 'custom_result'

            def create(self, model):
                return []

        return MockInterfaceCustom

    @pytest.fixture
    def factory_single_interface(self, clear, mock_interface1):
        """Create factory with single interface"""
        return InterfaceFactoryTemplate([mock_interface1])

    @pytest.fixture
    def factory_multiple_interfaces(self, clear, mock_interface1, mock_interface2):
        """Create factory with multiple interfaces"""
        return InterfaceFactoryTemplate([mock_interface1, mock_interface2])

    def test_init_with_single_interface_default_selection(self, clear, mock_interface1):
        """Test initialization with single interface uses first as default"""
        # When
        factory = InterfaceFactoryTemplate([mock_interface1])

        # Then
        assert factory._interfaces == [mock_interface1]
        assert factory._current_interface == mock_interface1
        assert factory() is not None
        assert isinstance(factory(), mock_interface1)

    def test_init_with_empty_interface_list_raises_error(self, clear):
        """Test initialization with empty interface list raises NotImplementedError"""
        # When/Then
        with pytest.raises(NotImplementedError):
            InterfaceFactoryTemplate([])

    def test_init_with_specific_interface_name(self, clear, mock_interface1, mock_interface2):
        """Test initialization with specific interface name"""
        # When
        factory = InterfaceFactoryTemplate(
            [mock_interface1, mock_interface2], interface_name='MockInterface2'
        )

        # Then
        assert factory._current_interface == mock_interface2
        assert isinstance(factory(), mock_interface2)

    def test_init_passes_args_and_kwargs_to_interface(self, clear, mock_interface1):
        """Test that args and kwargs are passed to interface constructor"""
        # When
        factory = InterfaceFactoryTemplate([mock_interface1], 'arg1', 'arg2', kwarg1='value1')

        # Then
        interface_obj = factory()
        assert interface_obj.args == ('arg1', 'arg2')
        assert interface_obj.kwargs == {'kwarg1': 'value1'}

    def test_create_with_interface_name(self, factory_multiple_interfaces, mock_interface2):
        """Test create method with specific interface name"""
        # When
        factory_multiple_interfaces.create(interface_name='MockInterface2')

        # Then
        assert factory_multiple_interfaces._current_interface == mock_interface2
        assert isinstance(factory_multiple_interfaces(), mock_interface2)

    def test_create_without_interface_name_uses_first(
        self, factory_multiple_interfaces, mock_interface1
    ):
        """Test create method without interface name uses first interface"""
        # When
        factory_multiple_interfaces.create()

        # Then
        assert factory_multiple_interfaces._current_interface == mock_interface1

    def test_available_interfaces_property(self, factory_multiple_interfaces):
        """Test available_interfaces property returns correct names"""
        # When
        interfaces = factory_multiple_interfaces.available_interfaces

        # Then
        assert 'MockInterface1' in interfaces
        assert 'MockInterface2' in interfaces
        assert len(interfaces) == 2

    def test_current_interface_property(self, factory_single_interface, mock_interface1):
        """Test current_interface property returns correct interface class"""
        # When/Then
        assert factory_single_interface.current_interface == mock_interface1

    def test_current_interface_name_property(self, factory_single_interface):
        """Test current_interface_name property returns correct name"""
        # When/Then
        assert factory_single_interface.current_interface_name == 'MockInterface1'

    def test_switch_to_valid_interface(self, factory_multiple_interfaces, mock_interface2):
        """Test switching to a valid interface"""
        # When
        factory_multiple_interfaces.switch('MockInterface2')

        # Then
        assert factory_multiple_interfaces._current_interface == mock_interface2
        assert isinstance(factory_multiple_interfaces(), mock_interface2)

    def test_switch_to_invalid_interface_raises_error(self, factory_single_interface):
        """Test switching to invalid interface raises AttributeError"""
        # When/Then
        with pytest.raises(AttributeError, match='The user supplied interface is not valid'):
            factory_single_interface.switch('NonExistentInterface')

    def test_switch_with_fitter_having_fit_object_with_update_bindings(
        self, factory_single_interface
    ):
        """Test switch with fitter that has _fit_object with update_bindings method"""
        # Given
        mock_fit_object = MagicMock()
        mock_fit_object.update_bindings = MagicMock()
        mock_fitter = MagicMock()
        mock_fitter._fit_object = mock_fit_object

        # When
        factory_single_interface.switch('MockInterface1', fitter=mock_fitter)

        # Then
        mock_fit_object.update_bindings.assert_called_once()

    def test_switch_with_fitter_having_generate_bindings(self, factory_single_interface):
        """Test switch with fitter that has generate_bindings method"""
        # Given
        mock_fitter = MagicMock()
        mock_fitter.generate_bindings = MagicMock()
        del mock_fitter._fit_object  # Ensure _fit_object doesn't exist

        # When
        factory_single_interface.switch('MockInterface1', fitter=mock_fitter)

        # Then
        mock_fitter.generate_bindings.assert_called_once()

    def test_switch_with_fitter_exception_handling(
        self, factory_single_interface, caplog: "pytest.LogCaptureFixture"
    ):
        """Test switch handles exceptions in fitter binding updates gracefully"""
        # Given
        mock_fit_object = MagicMock()
        mock_fit_object.update_bindings.side_effect = Exception('Test exception')
        mock_fitter = MagicMock()
        mock_fitter._fit_object = mock_fit_object

        # When
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.calculators'):
            factory_single_interface.switch('MockInterface1', fitter=mock_fitter)

        # Then
        assert 'Unable to auto generate bindings' in caplog.text
        assert 'Test exception' in caplog.text

    def test_fit_func_property_returns_callable(self, factory_single_interface):
        """Test fit_func property returns a callable"""
        # When
        fit_func = factory_single_interface.fit_func

        # Then
        assert callable(fit_func)

    def test_fit_func_calls_interface_fit_func(self, factory_single_interface):
        """Test fit_func calls the underlying interface's fit_func"""
        # When
        result = factory_single_interface.fit_func('arg1', kwarg1='value1')

        # Then
        assert result == 'result1'

    def test_call_method(self, factory_single_interface):
        """Test call method delegates to fit_func"""
        # When
        result = factory_single_interface.call('arg1', kwarg1='value1')

        # Then
        assert result == 'result1'

    def test_generate_bindings_with_matching_properties(self, factory_single_interface):
        """Test generate_bindings method with matching properties"""
        # Given
        mock_model = MagicMock()
        mock_prop = MagicMock()
        mock_prop.name = 'test_param'
        mock_prop.value = 42
        mock_model._get_linkable_attributes.return_value = [mock_prop]

        mock_item = MagicMock()
        mock_item.name_conversion = {'test_param': 'internal_param'}
        mock_item.make_prop.return_value = MagicMock()

        # Mock the interface object's create method
        interface_obj = factory_single_interface()
        interface_obj.create = MagicMock(return_value=[mock_item])

        # When
        factory_single_interface.generate_bindings(mock_model)

        # Then
        mock_item.make_prop.assert_called_once_with('test_param')
        assert mock_prop._callback is not None

    def test_generate_bindings_with_value_no_call_back_property(self, factory_single_interface):
        """Test generate_bindings handles properties with value_no_call_back"""
        # Given
        mock_model = MagicMock()
        mock_prop = MagicMock()
        mock_prop.name = 'test_param'
        mock_prop.value_no_call_back = 24
        mock_model._get_linkable_attributes.return_value = [mock_prop]

        mock_item = MagicMock()
        mock_item.name_conversion = {'test_param': 'internal_param'}
        mock_callback = MagicMock()
        mock_item.make_prop.return_value = mock_callback

        # Mock the interface object's create method
        interface_obj = factory_single_interface()
        interface_obj.create = MagicMock(return_value=[mock_item])

        # When
        factory_single_interface.generate_bindings(mock_model)

        # Then
        mock_callback.fset.assert_called_once_with(24)

    def test_generate_bindings_skips_non_matching_properties(self, factory_single_interface):
        """Test generate_bindings skips properties not in name_conversion"""
        # Given
        mock_model = MagicMock()
        mock_prop = MagicMock()
        mock_prop.name = 'non_matching_param'
        mock_model._get_linkable_attributes.return_value = [mock_prop]

        mock_item = MagicMock()
        mock_item.name_conversion = {'different_param': 'internal_param'}

        # Mock the interface object's create method
        interface_obj = factory_single_interface()
        interface_obj.create = MagicMock(return_value=[mock_item])

        # When
        factory_single_interface.generate_bindings(mock_model)

        # Then
        mock_item.make_prop.assert_not_called()

    def test_call_dunder_method_returns_interface_obj(self, factory_single_interface):
        """Test __call__ method returns the interface object"""
        # When
        result1 = factory_single_interface()
        result2 = factory_single_interface()

        # Then
        assert result1 is not None
        assert result1 is result2  # Should return the same object

    def test_reduce_method_for_pickle_support(self, factory_single_interface):
        """Test __reduce__ method for pickle serialization support"""
        # When
        result = factory_single_interface.__reduce__()

        # Then
        assert len(result) == 2
        assert result[0] == InterfaceFactoryTemplate.__state_restore__
        assert result[1][0] == InterfaceFactoryTemplate
        assert result[1][1] == 'MockInterface1'

    def test_state_restore_static_method_implementation_issue(
        self, clear, mock_interface1, mock_interface2
    ):
        """Test __state_restore__ static method - demonstrates implementation issue"""
        # Given
        original_factory = InterfaceFactoryTemplate([mock_interface1, mock_interface2])

        # When/Then - this implementation has a bug where it tries to call cls() without arguments
        # which would fail since the constructor requires interface_list parameter
        with pytest.raises(TypeError):
            InterfaceFactoryTemplate.__state_restore__(InterfaceFactoryTemplate, 'MockInterface2')

    def test_return_name_static_method_with_name_attribute(self, mock_interface_with_custom_name):
        """Test return_name static method with interface having name attribute"""
        # When
        name = InterfaceFactoryTemplate.return_name(mock_interface_with_custom_name)

        # Then
        assert name == 'CustomName'

    def test_return_name_static_method_without_name_attribute(self, mock_interface1):
        """Test return_name static method without name attribute uses __name__"""
        # When
        name = InterfaceFactoryTemplate.return_name(mock_interface1)

        # Then
        assert name == 'MockInterface1'

    def test_switch_with_fitter_generate_bindings_exception(
        self, factory_single_interface, caplog: "pytest.LogCaptureFixture"
    ):
        """Test switch handles exceptions in fitter.generate_bindings gracefully"""
        # Given
        mock_fitter = MagicMock()
        mock_fitter.generate_bindings.side_effect = Exception('Generate bindings test exception')
        # Ensure _fit_object doesn't exist so it goes to the elif branch
        if hasattr(mock_fitter, '_fit_object'):
            del mock_fitter._fit_object

        # When
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.calculators'):
            factory_single_interface.switch('MockInterface1', fitter=mock_fitter)

        # Then
        assert 'Unable to auto generate bindings' in caplog.text
        assert 'Generate bindings test exception' in caplog.text

    def test_generate_bindings_with_property_without_value_no_call_back(
        self, factory_single_interface
    ):
        """Test generate_bindings handles properties without value_no_call_back (descriptor objects)"""
        # Given
        mock_model = MagicMock()
        mock_prop = MagicMock()
        mock_prop.name = 'test_param'
        mock_prop.value = 99  # This will be used since no value_no_call_back attribute
        # Explicitly remove value_no_call_back to force the else branch
        del mock_prop.value_no_call_back
        mock_model._get_linkable_attributes.return_value = [mock_prop]

        mock_item = MagicMock()
        mock_item.name_conversion = {'test_param': 'internal_param'}
        mock_callback = MagicMock()
        mock_item.make_prop.return_value = mock_callback

        interface_obj = factory_single_interface()
        interface_obj.create = MagicMock(return_value=[mock_item])

        # When
        factory_single_interface.generate_bindings(mock_model)

        # Then
        mock_callback.fset.assert_called_once_with(99)


class TestItemContainer:
    @pytest.fixture
    def mock_getter(self):
        """Mock getter function"""

        def getter(link_name, key):
            return f'got_{link_name}_{key}'

        return getter

    @pytest.fixture
    def mock_setter(self):
        """Mock setter function"""

        def setter(link_name, **kwargs):
            pass

        return setter

    @pytest.fixture
    def item_container(self, mock_getter, mock_setter):
        """Create ItemContainer instance"""
        return ItemContainer(
            link_name='test_link',
            name_conversion={'param1': 'internal_param1', 'param2': 'internal_param2'},
            getter_fn=mock_getter,
            setter_fn=mock_setter,
        )

    def test_item_container_creation(self, item_container):
        """Test ItemContainer can be created with required fields"""
        # Then
        assert item_container.link_name == 'test_link'
        assert item_container.name_conversion == {
            'param1': 'internal_param1',
            'param2': 'internal_param2',
        }
        assert callable(item_container.getter_fn)
        assert callable(item_container.setter_fn)

    def test_convert_key_with_existing_key(self, item_container):
        """Test convert_key method with existing key"""
        # When
        result = item_container.convert_key('param1')

        # Then
        assert result == 'internal_param1'

    def test_convert_key_with_non_existing_key(self, item_container):
        """Test convert_key method with non-existing key returns None"""
        # When
        result = item_container.convert_key('non_existing')

        # Then
        assert result is None

    def test_make_prop_returns_property(self, item_container):
        """Test make_prop method returns a property object"""
        # When
        prop = item_container.make_prop('param1')

        # Then
        assert isinstance(prop, property)
        assert prop.fget is not None
        assert prop.fset is not None

    def test_make_getter_function_behavior(self, item_container):
        """Test the behavior of the __make_getter method indirectly"""
        # When
        prop = item_container.make_prop('param1')

        # Then
        assert isinstance(prop, property)
        assert prop.fget is not None
        assert callable(prop.fget)

    def test_make_setter_function_behavior(self, item_container):
        """Test the behavior of the __make_setter method indirectly"""
        # Given
        setter_calls = []

        def mock_setter(link_name, **kwargs):
            setter_calls.append((link_name, kwargs))

        container = ItemContainer(
            link_name='test_link',
            name_conversion={'param1': 'internal_param1'},
            getter_fn=lambda x, y: None,
            setter_fn=mock_setter,
        )

        # When
        prop = container.make_prop('param1')

        # Then
        assert isinstance(prop, property)
        assert prop.fset is not None
        assert callable(prop.fset)

    def test_convert_key_behavior_in_property_creation(self, item_container):
        """Test that convert_key behavior is used correctly in property creation"""
        # When
        prop1 = item_container.make_prop('param1')  # exists in name_conversion
        prop2 = item_container.make_prop('non_existing_param')  # doesn't exist

        # Then
        # Both should create valid property objects
        assert isinstance(prop1, property)
        assert isinstance(prop2, property)
        # The difference is in the internal key mapping which affects runtime behavior

    def test_actual_getter_setter_execution(self):
        """Test that getter and setter functions are actually executed"""
        # Given
        getter_calls = []
        setter_calls = []

        def mock_getter(link_name, key):
            getter_calls.append((link_name, key))
            return f'value_for_{key}'

        def mock_setter(link_name, **kwargs):
            setter_calls.append((link_name, kwargs))

        container = ItemContainer(
            link_name='test_link',
            name_conversion={'param1': 'internal_param1'},
            getter_fn=mock_getter,
            setter_fn=mock_setter,
        )

        # When
        prop = container.make_prop('param1')

        # Execute the getter function directly (this covers the actual getter execution)
        getter_func = prop.fget
        getter_result = getter_func()  # type: ignore

        # Execute the setter function directly (this covers the actual setter execution)
        setter_func = prop.fset
        setter_func('test_value')  # type: ignore

        # Then
        assert getter_result == 'value_for_internal_param1'
        assert len(getter_calls) == 1
        assert getter_calls[0] == ('test_link', 'internal_param1')

        assert len(setter_calls) == 1
        assert setter_calls[0] == ('test_link', {'internal_param1': 'test_value'})
