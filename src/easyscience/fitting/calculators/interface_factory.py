# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Type

if TYPE_CHECKING:
    from abc import ABCMeta

    from .. import Fitter


class InterfaceFactoryTemplate:
    """
    This class allows for the creation and transference of interfaces.
    """

    def __init__(self, interface_list: List[ABCMeta], *args, **kwargs):
        self._interfaces: List[ABCMeta] = interface_list
        self._current_interface: ABCMeta
        self.__interface_obj: None = None
        self.create(*args, **kwargs)

    def create(self, *args: Any, interface_name: str | None = None, **kwargs: Any) -> None:
        """
        Create an interface to a calculator from those initialized.

        Interfaces can be selected by ``interface_name`` where
        ``interface_name`` is one of ``obj.available_interfaces``. This
        interface can now be accessed by obj().

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to the interface constructor.
        interface_name : str | None, default=None
            Name of interface to be created.

        **kwargs : Any
            Keyword arguments forwarded to the interface constructor.

        Raises
        ------
        NotImplementedError
            If no interfaces are available to instantiate.
        """
        if interface_name is None:
            if len(self._interfaces) > 0:
                # Fallback name
                interface_name = self.return_name(self._interfaces[0])
            else:
                raise NotImplementedError
        interfaces = self.available_interfaces
        if interface_name in interfaces:
            self._current_interface = self._interfaces[interfaces.index(interface_name)]
        self.__interface_obj = self._current_interface(*args, **kwargs)

    def switch(self, new_interface: str, fitter: Optional[Type[Fitter]] = None) -> None:
        """
        Changes the current interface to a new interface.

        The current interface is destroyed and all SerializerComponent
        parameters carried over to the new interface. i.e. pick up where
        you left off.

        Parameters
        ----------
        new_interface : str
            Name of new interface to be created.
        fitter : Optional[Type[Fitter]], default=None
            Fitting interface which contains the fitting object which
            may have bindings which will be updated. By default, None.

        Raises
        ------
        AttributeError
            If ``new_interface`` is not a valid interface name.
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

        Returns
        -------
        List[str]
            List of available interface names.
        """
        return [self.return_name(this_interface) for this_interface in self._interfaces]

    @property
    def current_interface(self) -> ABCMeta:
        """
        Returns the constructor for the currently selected interface.

        Returns
        -------
        ABCMeta
            Interface constructor.
        """
        return self._current_interface

    @property
    def current_interface_name(self) -> str:
        """
        Returns the constructor name for the currently selected
        interface.

        Returns
        -------
        str
            Interface constructor name.
        """
        return self.return_name(self._current_interface)

    @property
    def fit_func(
        self,
    ) -> Callable:  # , x_array: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Pass through to the underlying interfaces fitting function.

        Returns
        -------
        Callable
            Callable proxy to the underlying interface fit function.
        """

        def __fit_func(*args, **kwargs):
            """Fit func."""
            return self.__interface_obj.fit_func(*args, **kwargs)

        return __fit_func

    def call(self, *args, **kwargs):
        """Call function."""
        return self.fit_func(*args, **kwargs)

    def generate_bindings(self, model: Any, *args: Any, ifun: Any = None, **kwargs: Any) -> None:
        """
        Automatically bind a ``Parameter`` to the corresponding
        interface.

        Parameters
        ----------
        model : Any
            Model whose linkable attributes should be bound.
        *args : Any
            Positional arguments reserved for interface-specific binding
            hooks.
        ifun : Any, default=None
            Optional interface hook. By default, None.
        **kwargs : Any
            Keyword arguments reserved for interface-specific binding
            hooks.
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

    def __call__(self, *args, **kwargs) -> None:
        """Call function."""
        return self.__interface_obj

    def __reduce__(self):
        """Reduce function."""
        return (
            self.__state_restore__,
            (
                self.__class__,
                self.current_interface_name,
            ),
        )

    @staticmethod
    def __state_restore__(cls, interface_str):
        """State restore."""
        obj = cls()
        if interface_str in obj.available_interfaces:
            obj.switch(interface_str)
        return obj

    @staticmethod
    def return_name(this_interface) -> str:
        """Return an interfaces name."""
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
        """Make prop."""
        return property(
            fget=self.__make_getter(parameter_name),
            fset=self.__make_setter(parameter_name),
        )

    def convert_key(self, lookup_key: str) -> str:
        """Convert key."""
        key = self.name_conversion.get(lookup_key, None)
        return key

    def __make_getter(self, get_name: str) -> Callable:
        """Make getter."""

        def get_value():
            """Get value."""
            inner_key = self.name_conversion.get(get_name, None)
            return self.getter_fn(self.link_name, inner_key)

        return get_value

    def __make_setter(self, get_name: str) -> Callable:
        """Make setter."""

        def set_value(value):
            """Set value."""
            inner_key = self.name_conversion.get(get_name, None)
            self.setter_fn(self.link_name, **{inner_key: value})

        return set_value
