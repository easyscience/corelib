# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easyscience import global_object
from easyscience.base_classes import EasyList
from easyscience.base_classes import NewBase
from easyscience.io import SerializerComponent
from easyscience.variable import DescriptorNumber
from easyscience.variable.descriptor_base import DescriptorBase


class TestDesciptorBase:
    @pytest.fixture
    def descriptor(self):
        # This avoids the error: TypeError: Can't instantiate abstract class DescriptorBase with abstract methods __init__
        DescriptorBase.__abstractmethods__ = set()
        DescriptorBase.__repr__ = lambda x: 'DescriptorBase'
        self.objs_before_new_descriptor = len(global_object.map.created_objs)
        descriptor = DescriptorBase(
            name='name',
            description='description',
            url='url',
            display_name='display_name',
            parent=None,
        )
        return descriptor

    @pytest.fixture
    def clear(self):
        global_object.map._clear()

    @pytest.mark.parametrize(
        'name',
        [1, True, 1.0, [], {}, (), None, object()],
        ids=['int', 'bool', 'float', 'list', 'dict', 'tuple', 'None', 'object'],
    )
    def test_init_name_type_error(self, name):
        # When Then
        with pytest.raises(TypeError):
            DescriptorBase(
                name=name,
                description='description',
                url='url',
                display_name='display_name',
                parent=None,
            )

    @pytest.mark.parametrize(
        'display_name',
        [1, True, 1.0, [], {}, (), object()],
        ids=['int', 'bool', 'float', 'list', 'dict', 'tuple', 'object'],
    )
    def test_init_display_name_type_error(self, display_name):
        # When Then
        with pytest.raises(TypeError):
            DescriptorBase(
                name='name',
                description='description',
                url='url',
                display_name=display_name,
                parent=None,
            )

    @pytest.mark.parametrize(
        'description',
        [1, True, 1.0, [], {}, (), object()],
        ids=['int', 'bool', 'float', 'list', 'dict', 'tuple', 'object'],
    )
    def test_init_description_type_error(self, description):
        # When Then
        with pytest.raises(TypeError):
            DescriptorBase(
                name='name',
                description=description,
                url='url',
                display_name='display_name',
                parent=None,
            )

    @pytest.mark.parametrize(
        'url',
        [1, True, 1.0, [], {}, (), object()],
        ids=['int', 'bool', 'float', 'list', 'dict', 'tuple', 'object'],
    )
    def test_init_url_type_error(self, url):
        # When Then
        with pytest.raises(TypeError):
            DescriptorBase(
                name='name',
                description='description',
                url=url,
                display_name='display_name',
                parent=None,
            )

    def test_init(self, descriptor: DescriptorBase):
        assert descriptor._name == 'name'
        assert descriptor._description == 'description'
        assert descriptor._url == 'url'
        assert descriptor._display_name == 'display_name'
        assert len(global_object.map.created_objs) - self.objs_before_new_descriptor == 1

    def test_display_name(self, descriptor: DescriptorBase):
        # When Then Expect
        assert descriptor.display_name == 'display_name'

    def test_display_name_none(self, descriptor: DescriptorBase):
        # When
        descriptor._display_name = None
        # Then Expect
        assert descriptor.display_name == 'name'

    def test_display_name_setter(self, descriptor: DescriptorBase):
        # When
        descriptor.display_name = 'new_display_name'
        # Then Expect
        assert descriptor.display_name == 'new_display_name'

    @pytest.mark.parametrize(
        'display_name',
        [1, True, 1.0, [], {}, (), object()],
        ids=['int', 'bool', 'float', 'list', 'dict', 'tuple', 'object'],
    )
    def test_display_name_setter_type_error(self, descriptor: DescriptorBase, display_name):
        # When Then
        with pytest.raises(TypeError):
            descriptor.display_name = display_name

    def test_name_setter(self, descriptor: DescriptorBase):
        # When
        descriptor.name = 'new_name'
        # Then Expect
        assert descriptor.name == 'new_name'

    @pytest.mark.parametrize(
        'name',
        [1, True, 1.0, [], {}, (), object(), None],
        ids=['int', 'bool', 'float', 'list', 'dict', 'tuple', 'object', 'None'],
    )
    def test_name_setter_type_error(self, descriptor: DescriptorBase, name):
        # When Then
        with pytest.raises(TypeError):
            descriptor.name = name

    def test_description_setter(self, descriptor: DescriptorBase):
        # When
        descriptor.description = 'new_description'
        # Then Expect
        assert descriptor.description == 'new_description'

    @pytest.mark.parametrize(
        'description',
        [1, True, 1.0, [], {}, (), object()],
        ids=['int', 'bool', 'float', 'list', 'dict', 'tuple', 'object'],
    )
    def test_description_setter_type_error(self, descriptor: DescriptorBase, description):
        # When Then
        with pytest.raises(TypeError):
            descriptor.description = description

    def test_url_setter(self, descriptor: DescriptorBase):
        # When
        descriptor.url = 'new_url'
        # Then Expect
        assert descriptor.url == 'new_url'

    @pytest.mark.parametrize(
        'url',
        [1, True, 1.0, [], {}, (), object()],
        ids=['int', 'bool', 'float', 'list', 'dict', 'tuple', 'object'],
    )
    def test_url_setter_type_error(self, descriptor: DescriptorBase, url):
        # When Then
        with pytest.raises(TypeError):
            descriptor.url = url

    @pytest.mark.parametrize('stack_enabled,stack_elements', [(False, 0), (True, 1)])
    def test_set_display_name_without_global_object_stack(
        self, descriptor: DescriptorBase, stack_enabled, stack_elements
    ):
        # When
        descriptor.__repr__ = lambda x: 'DescriptorBase'
        global_object.stack.clear()
        global_object.stack._enabled = stack_enabled

        # Then
        descriptor.display_name = 'new_display_name'

        # Expect
        assert descriptor.display_name == 'new_display_name'
        assert len(global_object.stack.history) == stack_elements

    def test_copy(self, descriptor: DescriptorBase):
        # When Then
        descriptor_copy = descriptor.__copy__()

        # Expect
        assert type(descriptor_copy) == DescriptorBase
        assert descriptor_copy._name == descriptor._name
        assert descriptor_copy._description == descriptor._description
        assert descriptor_copy._url == descriptor._url
        assert descriptor_copy._display_name == descriptor._display_name

    def test_unique_name_generator(self, clear, descriptor: DescriptorBase):
        # When
        second_descriptor = DescriptorBase(name='test', unique_name='DescriptorBase_2')

        # Then
        third_descriptor = DescriptorBase(name='test2')
        fourth_descriptor = DescriptorBase(name='test3')

        # Expect
        assert descriptor.unique_name == 'DescriptorBase_0'
        assert third_descriptor.unique_name == 'DescriptorBase_3'
        assert fourth_descriptor.unique_name == 'DescriptorBase_4'

    def test_unique_name_change(self, clear, descriptor: DescriptorBase):
        # When Then
        descriptor.unique_name = 'test'
        # Expect
        assert descriptor.unique_name == 'test'

    @pytest.mark.parametrize('input', [2, 2.0, [2], {2}, None])
    def test_unique_name_change_exception(self, input, descriptor: DescriptorBase):
        # When Then Expect
        with pytest.raises(TypeError):
            descriptor.unique_name = input

    def test_is_a_new_base(self, descriptor: DescriptorBase):
        # When Then Expect
        assert isinstance(descriptor, NewBase)
        assert not isinstance(descriptor, SerializerComponent)

    def test_as_dict_is_an_alias_of_to_dict(self, descriptor: DescriptorBase):
        # When Then Expect
        assert descriptor.as_dict() == descriptor.to_dict()
        assert descriptor.as_dict(skip=['url']) == descriptor.to_dict(skip=['url'])

    def test_to_dict_drops_generated_unique_name(self, clear):
        """Descriptors follow the ``NewBase`` design: an auto-generated
        unique_name is not serialized, so a decoded descriptor is given
        a fresh one instead of colliding with the original."""
        # When
        descriptor = DescriptorNumber(name='name', value=1.0)

        # Then Expect
        assert descriptor._default_unique_name
        assert 'unique_name' not in descriptor.to_dict()

    def test_to_dict_keeps_explicit_unique_name(self, clear):
        """An explicitly supplied unique_name is still serialized."""
        # When
        descriptor = DescriptorNumber(name='name', value=1.0, unique_name='explicit_name')

        # Then Expect
        assert not descriptor._default_unique_name
        assert descriptor.to_dict()['unique_name'] == 'explicit_name'

    def test_can_be_held_by_an_easy_list(self, clear):
        """Descriptors are NewBase objects, so EasyList accepts them."""
        # When
        descriptor = DescriptorNumber(name='name', value=1.0)
        easy_list = EasyList(descriptor)

        # Then Expect
        assert list(easy_list) == [descriptor]
        assert easy_list[descriptor.unique_name] is descriptor
