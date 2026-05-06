# SPDX-FileCopyrightText: 2025 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from typing import Callable
from typing import Optional

from ..base_classes.based_base import BasedBase
from ..utils.classTools import addLoggedProp
from ..variable.descriptor_base import DescriptorBase

if TYPE_CHECKING:
    from ..io import SerializerComponent


class ObjBase(BasedBase):
    """This is the base class for which all higher level classes are
    built off of.

    .. deprecated::
        `ObjBase` is deprecated and will be removed in a future version.
        Please migrate to `ModelBase` instead.

    NOTE: This object is serializable only if parameters are supplied as:
    `ObjBase(a=value, b=value)`. For `Parameter` or `Descriptor` objects we can
    cheat with `ObjBase(*[Descriptor(...), Parameter(...), ...])`.
    """

    def __init__(
        self,
        name: str,
        unique_name: Optional[str] = None,
        *args: Optional[SerializerComponent],
        **kwargs: Optional[SerializerComponent],
    ):
        """Set up the base class.

        :param name: Name of this object
        :param args: Any arguments?
        :param kwargs: Fields which this class should contain
        """
        warnings.warn(
            'ObjBase is deprecated and will be removed in a future version. '
            'Please migrate to ModelBase.',
            DeprecationWarning,
            stacklevel=2,
        )
        super(ObjBase, self).__init__(name=name, unique_name=unique_name)
        # If Parameter or Descriptor is given as arguments...
        for arg in args:
            if issubclass(type(arg), (ObjBase, DescriptorBase)):
                kwargs[getattr(arg, 'name')] = arg
        # Set kwargs, also useful for serialization
        known_keys = self.__dict__.keys()
        self._kwargs = kwargs
        for key in kwargs.keys():
            if key in known_keys:
                raise AttributeError('Kwargs cannot overwrite class attributes in ObjBase.')
            if issubclass(type(kwargs[key]), (BasedBase, DescriptorBase)) or 'CollectionBase' in [
                c.__name__ for c in type(kwargs[key]).__bases__
            ]:
                self._global_object.map.add_edge(self, kwargs[key])
                self._global_object.map.reset_type(kwargs[key], 'created_internal')
            addLoggedProp(
                self,
                key,
                self.__getter(key),
                self.__setter(key),
                get_id=key,
                my_self=self,
                test_class=ObjBase,
            )

    def _add_component(self, key: str, component: SerializerComponent) -> None:
        """Dynamically add a component to the class. This is an internal
        method, though can be called remotely. The recommended
        alternative is to use typing, i.e.

        .. code-block:: python

            class Foo(Bar):
                def __init__(self, foo: Parameter, bar: Parameter):
                    super(Foo, self).__init__(bar=bar)
                    self._add_component('foo', foo)

        :param key: Name of component to be added
        :param component: Component to be added
        :return: None
        """
        self._kwargs[key] = component
        self._global_object.map.add_edge(self, component)
        self._global_object.map.reset_type(component, 'created_internal')
        addLoggedProp(
            self,
            key,
            self.__getter(key),
            self.__setter(key),
            get_id=key,
            my_self=self,
            test_class=ObjBase,
        )

    def __setattr__(self, key: str, value: SerializerComponent) -> None:
        # Assume that the annotation is a ClassVar
        old_obj = None
        if (
            hasattr(self.__class__, '__annotations__')
            and key in self.__class__.__annotations__
            and hasattr(self.__class__.__annotations__[key], '__args__')
            and issubclass(
                getattr(value, '__old_class__', value.__class__),
                self.__class__.__annotations__[key].__args__,
            )
        ):
            if issubclass(type(getattr(self, key, None)), (BasedBase, DescriptorBase)):
                old_obj = self.__getattribute__(key)
                self._global_object.map.prune_vertex_from_edge(self, old_obj)
            self._add_component(key, value)
        else:
            if hasattr(self, key) and issubclass(type(value), (BasedBase, DescriptorBase)):
                old_obj = self.__getattribute__(key)
                self._global_object.map.prune_vertex_from_edge(self, old_obj)
                self._global_object.map.add_edge(self, value)
        super(ObjBase, self).__setattr__(key, value)
        # Update the interface bindings if something changed (BasedBase and Descriptor)
        if old_obj is not None:
            old_interface = getattr(self, 'interface', None)
            if old_interface is not None:
                self.generate_bindings()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} `{getattr(self, "name")}`'

    @staticmethod
    def __getter(key: str) -> Callable[[SerializerComponent], SerializerComponent]:
        def getter(obj: SerializerComponent) -> SerializerComponent:
            return obj._kwargs[key]

        return getter

    @staticmethod
    def __setter(key: str) -> Callable[[SerializerComponent], None]:
        def setter(obj: SerializerComponent, value: float) -> None:
            if issubclass(obj._kwargs[key].__class__, (DescriptorBase)) and not issubclass(
                value.__class__, (DescriptorBase)
            ):
                obj._kwargs[key].value = value
            else:
                obj._kwargs[key] = value

        return setter
