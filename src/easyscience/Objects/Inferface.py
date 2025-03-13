from __future__ import annotations

#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

from abc import ABCMeta
from typing import TYPE_CHECKING
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Type
from typing import TypeVar

_C = TypeVar('_C', bound=ABCMeta)
_M = TypeVar('_M')
if TYPE_CHECKING:
    from easyscience.fitting import Fitter


class InterfaceFactoryTemplate:
    """
    This class allows for the creation and transference of interfaces.
    """

    def __init__(self, interface_list: List[_C], *args, **kwargs):
        self._interfaces: List[_C] = interface_list
        self._current_interface: _C
        self.__interface_obj: _M = None
        self.create(*args, **kwargs)

    def create(self, *args, **kwargs):
        """
        Create an interface to a calculator from those initialized. Interfaces can be selected
        by `interface_name` where `interface_name` is one of `obj.available_interfaces`. This
        interface can now be accessed by obj().

        :param interface_name: name of interface to be created
        :type interface_name: str
        :return: None
        :rtype: noneType
        """
        if kwargs.get('interface_name', None) is None:
            if len(self._interfaces) > 0:
                # Fallback name
                interface_name = self.return_name(self._interfaces[0])
            else:
                raise NotImplementedError
        else:
            interface_name = kwargs.pop('interface_name')
        interfaces = self.available_interfaces
        if interface_name in interfaces:
            self._current_interface = self._interfaces[interfaces.index(interface_name)]
        self.__interface_obj = self._current_interface(*args, **kwargs)

    def switch(self, new_interface: str, fitter: Optional[Type[Fitter]] = None):
        """
        Changes the current interface to a new interface. The current interface is destroyed and
        all ComponentSerializer parameters carried over to the new interface. i.e. pick up where you left off.

        :param new_interface: name of new interface to be created
        :type new_interface: str
        :param fitter: Fitting interface which contains the fitting object which may have bindings which will be updated.
        :type fitter: EasyScience.fitting.Fitter
        :return: None
        :rtype: noneType
        """
        interfaces = self.available_interfaces
        if new_interface in interfaces:
            self._current_interface = self._interfaces[interfaces.index(new_interface)]
            self.__interface_obj = self._current_interface()
        else:
            raise AttributeError('The user supplied interface is not valid.')
        if fitter is not None:
            if hasattr(fitter, '_fit_object'):
                obj = getattr(fitter, '_fit_object')
                try:
                    if hasattr(obj, 'update_bindings'):
                        obj.update_bindings()
                except Exception as e:
                    print(f'Unable to auto generate bindings.\n{e}')
            elif hasattr(fitter, 'generate_bindings'):
                try:
                    fitter.generate_bindings()
                except Exception as e:
                    print(f'Unable to auto generate bindings.\n{e}')

    @property
    def available_interfaces(self) -> List[str]:
        """
        Return all available interfaces.

        :return: List of available interface names
        :rtype: List[str]
        """
        return [self.return_name(this_interface) for this_interface in self._interfaces]

    @property
    def current_interface(self) -> _C:
        """
        Returns the constructor for the currently selected interface.

        :return: Interface constructor
        :rtype: InterfaceTemplate
        """
        return self._current_interface

    @property
    def current_interface_name(self) -> str:
        """
        Returns the constructor name for the currently selected interface.

        :return: Interface constructor name
        :rtype: str
        """
        return self.return_name(self._current_interface)

    @property
    def fit_func(
        self,
    ) -> Callable:  # , x_array: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Pass through to the underlying interfaces fitting function.

        :param x_array: points to be calculated at
        :type x_array: np.ndarray
        :param args: positional arguments for the fitting function
        :type args: Any
        :param kwargs: key/value pair arguments for the fitting function.
        :type kwargs: Any
        :return: points calculated at positional values `x`
        :rtype: np.ndarray
        #"""

        def __fit_func(*args, **kwargs):
            return self.__interface_obj.fit_func(*args, **kwargs)

        return __fit_func

    def call(self, *args, **kwargs):
        return self.fit_func(*args, **kwargs)

    def generate_bindings(self, model, *args, ifun=None, **kwargs):
        """
        Automatically bind a `Parameter` to the corresponding interface.
        :param name: parameter name
        :type name: str
        :return: binding property
        :rtype: property
        """

        class_links = self.__interface_obj.create(model)
        props = model._get_linkable_attributes()
        props_names = [prop.name for prop in props]
        for item in class_links:
            for item_key in item.name_conversion.keys():
                if item_key not in props_names:
                    continue
                idx = props_names.index(item_key)
                prop = props[idx]

                # Should be fetched this way to ensure we don't get value from callback
                if hasattr(prop, 'value_no_call_back'):
                    # Property object
                    prop_value = prop.value_no_call_back
                else:
                    # Descriptor object
                    prop_value = prop.value

                prop._callback = item.make_prop(item_key)
                prop._callback.fset(prop_value)

    def __call__(self, *args, **kwargs) -> _M:
        return self.__interface_obj

    def __reduce__(self):
        return (
            self.__state_restore__,
            (
                self.__class__,
                self.current_interface_name,
            ),
        )

    @staticmethod
    def __state_restore__(cls, interface_str):
        obj = cls()
        if interface_str in obj.available_interfaces:
            obj.switch(interface_str)
        return obj

    @staticmethod
    def return_name(this_interface) -> str:
        """
        Return an interfaces name
        """
        interface_name = this_interface.__name__
        if hasattr(this_interface, 'name'):
            interface_name = getattr(this_interface, 'name')
        return interface_name


class ItemContainer(NamedTuple):
    link_name: str
    name_conversion: dict
    getter_fn: Callable
    setter_fn: Callable

    def make_prop(self, parameter_name) -> property:
        return property(
            fget=self.__make_getter(parameter_name),
            fset=self.__make_setter(parameter_name),
        )

    def convert_key(self, lookup_key: str) -> str:
        key = self.name_conversion.get(lookup_key, None)
        return key

    def __make_getter(self, get_name: str) -> Callable:
        def get_value():
            inner_key = self.name_conversion.get(get_name, None)
            return self.getter_fn(self.link_name, inner_key)

        return get_value

    def __make_setter(self, get_name: str) -> Callable:
        def set_value(value):
            inner_key = self.name_conversion.get(get_name, None)
            self.setter_fn(self.link_name, **{inner_key: value})

        return set_value


iF = TypeVar('iF', bound=InterfaceFactoryTemplate)
