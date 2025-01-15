from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from inspect import getfullargspec
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import TypeVar

from easyscience import global_object
from easyscience.Utils.classTools import addLoggedProp

from .core import ComponentSerializer
from .variable import Parameter
from .variable.descriptor_base import DescriptorBase

if TYPE_CHECKING:
    from easyscience.Constraints import C
    from easyscience.Objects.Inferface import iF
    V = TypeVar('V', bound=DescriptorBase)


class BasedBase(ComponentSerializer):
    __slots__ = ['_name', '_global_object', 'user_data', '_kwargs']

    _REDIRECT = {}

    def __init__(self, name: str, interface: Optional[iF] = None, unique_name: Optional[str] = None):
        self._global_object = global_object
        if unique_name is None:
            unique_name = self._global_object.generate_unique_name(self.__class__.__name__)
        self._unique_name = unique_name
        self._name = name
        self._global_object.map.add_vertex(self, obj_type='created')
        self.interface = interface
        self.user_data: dict = {}

    @property
    def _arg_spec(self) -> Set[str]:
        base_cls = getattr(self, '__old_class__', self.__class__)
        spec = getfullargspec(base_cls.__init__)
        names = set(spec.args[1:])
        return names

    def __reduce__(self):
        """
        Make the class picklable.
        Due to the nature of the dynamic class definitions special measures need to be taken.

        :return: Tuple consisting of how to make the object
        :rtype: tuple
        """
        state = self.encode()
        cls = getattr(self, '__old_class__', self.__class__)
        return cls.from_dict, (state,)

    @property
    def unique_name(self) -> str:
        """Get the unique name of the object."""
        return self._unique_name

    @unique_name.setter
    def unique_name(self, new_unique_name: str):
        """Set a new unique name for the object. The old name is still kept in the map.

        :param new_unique_name: New unique name for the object"""
        if not isinstance(new_unique_name, str):
            raise TypeError('Unique name has to be a string.')
        self._unique_name = new_unique_name
        self._global_object.map.add_vertex(self)

    @property
    def name(self) -> str:
        """
        Get the common name of the object.

        :return: Common name of the object
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        """
        Set a new common name for the object.

        :param new_name: New name for the object
        :return: None
        """
        self._name = new_name

    @property
    def interface(self) -> iF:
        """
        Get the current interface of the object
        """
        return self._interface

    @interface.setter
    def interface(self, new_interface: iF):
        """
        Set the current interface to the object and generate bindings if possible. iF.e.
        ```
        def __init__(self, bar, interface=None, **kwargs):
            super().__init__(self, **kwargs)
            self.foo = bar
            self.interface = interface # As final step after initialization to set correct bindings.
        ```
        """
        self._interface = new_interface
        if new_interface is not None:
            self.generate_bindings()

    def generate_bindings(self):
        """
        Generate or re-generate bindings to an interface (if exists)

        :raises: AttributeError
        """
        if self.interface is None:
            raise AttributeError('Interface error for generating bindings. `interface` has to be set.')
        interfaceable_children = [
            key
            for key in self._global_object.map.get_edges(self)
            if issubclass(type(self._global_object.map.get_item_by_key(key)), BasedBase)
        ]
        for child_key in interfaceable_children:
            child = self._global_object.map.get_item_by_key(child_key)
            child.interface = self.interface
        self.interface.generate_bindings(self)

    def switch_interface(self, new_interface_name: str):
        """
        Switch or create a new interface.
        """
        if self.interface is None:
            raise AttributeError('Interface error for generating bindings. `interface` has to be set.')
        self.interface.switch(new_interface_name)
        self.generate_bindings()

    @property
    def constraints(self) -> List[C]:
        pars = self.get_parameters()
        constraints = []
        for par in pars:
            con: Dict[str, C] = par.user_constraints
            for key in con.keys():
                constraints.append(con[key])
        return constraints

    def get_parameters(self) -> List[Parameter]:
        """
        Get all parameter objects as a list.

        :return: List of `Parameter` objects.
        """
        par_list = []
        for key, item in self._kwargs.items():
            if hasattr(item, 'get_parameters'):
                par_list = [*par_list, *item.get_parameters()]
            elif isinstance(item, Parameter):
                par_list.append(item)
        return par_list

    def _get_linkable_attributes(self) -> List[V]:
        """
        Get all objects which can be linked against as a list.

        :return: List of `Descriptor`/`Parameter` objects.
        """
        item_list = []
        for key, item in self._kwargs.items():
            if hasattr(item, '_get_linkable_attributes'):
                item_list = [*item_list, *item._get_linkable_attributes()]
            elif issubclass(type(item), (DescriptorBase)):
                item_list.append(item)
        return item_list

    def get_fit_parameters(self) -> List[Parameter]:
        """
        Get all objects which can be fitted (and are not fixed) as a list.

        :return: List of `Parameter` objects which can be used in fitting.
        """
        fit_list = []
        for key, item in self._kwargs.items():
            if hasattr(item, 'get_fit_parameters'):
                fit_list = [*fit_list, *item.get_fit_parameters()]
            elif isinstance(item, Parameter):
                if item.enabled and not item.fixed:
                    fit_list.append(item)
        return fit_list

    def __dir__(self) -> Iterable[str]:
        """
        This creates auto-completion and helps out in iPython notebooks.

        :return: list of function and parameter names for auto-completion
        """
        new_class_objs = list(k for k in dir(self.__class__) if not k.startswith('_'))
        return sorted(new_class_objs)

    def __copy__(self) -> BasedBase:
        """Return a copy of the object."""
        temp = self.as_dict(skip=['unique_name'])
        new_obj = self.__class__.from_dict(temp)
        return new_obj


if TYPE_CHECKING:
    B = TypeVar('B', bound=BasedBase)
    BV = TypeVar('BV', bound=ComponentSerializer)


class BaseObj(BasedBase):
    """
    This is the base class for which all higher level classes are built off of.
    NOTE: This object is serializable only if parameters are supplied as:
    `BaseObj(a=value, b=value)`. For `Parameter` or `Descriptor` objects we can
    cheat with `BaseObj(*[Descriptor(...), Parameter(...), ...])`.
    """

    def __init__(
        self,
        name: str,
        unique_name: Optional[str] = None,
        *args: Optional[BV],
        **kwargs: Optional[BV],
    ):
        """
        Set up the base class.

        :param name: Name of this object
        :param args: Any arguments?
        :param kwargs: Fields which this class should contain
        """
        super(BaseObj, self).__init__(name=name, unique_name=unique_name)
        # If Parameter or Descriptor is given as arguments...
        for arg in args:
            if issubclass(type(arg), (BaseObj, DescriptorBase)):
                kwargs[getattr(arg, 'name')] = arg
        # Set kwargs, also useful for serialization
        known_keys = self.__dict__.keys()
        self._kwargs = kwargs
        for key in kwargs.keys():
            if key in known_keys:
                raise AttributeError('Kwargs cannot overwrite class attributes in BaseObj.')
            if issubclass(type(kwargs[key]), (BasedBase, DescriptorBase)) or 'BaseCollection' in [
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
                test_class=BaseObj,
            )

    def _add_component(self, key: str, component: BV) -> None:
        """
        Dynamically add a component to the class. This is an internal method, though can be called remotely.
        The recommended alternative is to use typing, i.e.

        class Foo(Bar):
            def __init__(self, foo: Parameter, bar: Parameter):
                super(Foo, self).__init__(bar=bar)
                self._add_component("foo", foo)

        Goes to:
         class Foo(Bar):
            foo: ClassVar[Parameter]
            def __init__(self, foo: Parameter, bar: Parameter):
                super(Foo, self).__init__(bar=bar)
                self.foo = foo

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
            test_class=BaseObj,
        )

    def __setattr__(self, key: str, value: BV) -> None:
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
        super(BaseObj, self).__setattr__(key, value)
        # Update the interface bindings if something changed (BasedBase and Descriptor)
        if old_obj is not None:
            old_interface = getattr(self, 'interface', None)
            if old_interface is not None:
                self.generate_bindings()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} `{getattr(self, 'name')}`"

    @staticmethod
    def __getter(key: str) -> Callable[[BV], BV]:
        def getter(obj: BV) -> BV:
            return obj._kwargs[key]

        return getter

    @staticmethod
    def __setter(key: str) -> Callable[[BV], None]:
        def setter(obj: BV, value: float) -> None:
            if issubclass(obj._kwargs[key].__class__, (DescriptorBase)) and not issubclass(
                value.__class__, (DescriptorBase)
            ):
                obj._kwargs[key].value = value
            else:
                obj._kwargs[key] = value

        return setter

    # @staticmethod
    # def __setter(key: str) -> Callable[[Union[B, V]], None]:
    #     def setter(obj: Union[V, B], value: float) -> None:
    #         if issubclass(obj._kwargs[key].__class__, Descriptor):
    #             if issubclass(obj._kwargs[key].__class__, Descriptor):
    #                 obj._kwargs[key] = value
    #             else:
    #                 obj._kwargs[key].value = value
    #         else:
    #             obj._kwargs[key] = value
    #
    #     return setter
