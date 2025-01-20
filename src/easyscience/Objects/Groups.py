#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

from collections.abc import MutableSequence
from numbers import Number
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

from easyscience.global_object.undo_redo import NotarizedDict
from easyscience.Objects.ObjectClasses import BasedBase
from easyscience.Objects.variable.descriptor_base import DescriptorBase

if TYPE_CHECKING:
    from easyscience.Objects.Inferface import iF
    from easyscience.Objects.ObjectClasses import B
    V = TypeVar('V', bound=DescriptorBase)


class BaseCollection(BasedBase, MutableSequence):
    """
    This is the base class for which all higher level classes are built off of.
    NOTE: This object is serializable only if parameters are supplied as:
    `BaseObj(a=value, b=value)`. For `Parameter` or `Descriptor` objects we can
    cheat with `BaseObj(*[Descriptor(...), Parameter(...), ...])`.
    """

    def __init__(
        self,
        name: str,
        *args: Union[B, V],
        interface: Optional[iF] = None,
        unique_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Set up the base collection class.

        :param name: Name of this object
        :type name: str
        :param args: selection of
        :param _kwargs: Fields which this class should contain
        :type _kwargs: dict
        """
        BasedBase.__init__(self, name, unique_name=unique_name)
        kwargs = {key: kwargs[key] for key in kwargs.keys() if kwargs[key] is not None}
        _args = []
        for item in args:
            if not isinstance(item, list):
                _args.append(item)
            else:
                _args += item
        _kwargs = {}
        for key, item in kwargs.items():
            if isinstance(item, list) and len(item) > 0:
                _args += item
            else:
                _kwargs[key] = item
        kwargs = _kwargs
        for item in list(kwargs.values()) + _args:
            if not issubclass(type(item), (DescriptorBase, BasedBase)):
                raise AttributeError('A collection can only be formed from easyscience objects.')
        args = _args
        _kwargs = {}
        for key, item in kwargs.items():
            _kwargs[key] = item
        for arg in args:
            kwargs[arg.unique_name] = arg
            _kwargs[arg.unique_name] = arg

        # Set kwargs, also useful for serialization
        self._kwargs = NotarizedDict(**_kwargs)

        for key in kwargs.keys():
            if key in self.__dict__.keys() or key in self.__slots__:
                raise AttributeError(f'Given kwarg: `{key}`, is an internal attribute. Please rename.')
            if kwargs[key]:  # Might be None (empty tuple or list)
                self._global_object.map.add_edge(self, kwargs[key])
                self._global_object.map.reset_type(kwargs[key], 'created_internal')
                if interface is not None:
                    kwargs[key].interface = interface
            # TODO wrap getter and setter in Logger
        if interface is not None:
            self.interface = interface
        self._kwargs._stack_enabled = True

    def insert(self, index: int, value: Union[V, B]) -> None:
        """
        Insert an object into the collection at an index.

        :param index: Index for EasyScience object to be inserted.
        :type index: int
        :param value: Object to be inserted.
        :type value: Union[BasedBase, DescriptorBase]
        :return: None
        :rtype: None
        """
        t_ = type(value)
        if issubclass(t_, (BasedBase, DescriptorBase)):
            update_key = list(self._kwargs.keys())
            values = list(self._kwargs.values())
            # Update the internal dict
            new_key = value.unique_name
            update_key.insert(index, new_key)
            values.insert(index, value)
            self._kwargs.reorder(**{k: v for k, v in zip(update_key, values)})
            # ADD EDGE
            self._global_object.map.add_edge(self, value)
            self._global_object.map.reset_type(value, 'created_internal')
            value.interface = self.interface
        else:
            raise AttributeError('Only EasyScience objects can be put into an EasyScience group')

    def __getitem__(self, idx: Union[int, slice]) -> Union[V, B]:
        """
        Get an item in the collection based on its index.

        :param idx: index or slice of the collection.
        :type idx: Union[int, slice]
        :return: Object at index `idx`
        :rtype: Union[Parameter, Descriptor, BaseObj, 'BaseCollection']
        """
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return self.__class__(getattr(self, 'name'), *[self[i] for i in range(start, stop, step)])
        if str(idx) in self._kwargs.keys():
            return self._kwargs[str(idx)]
        if isinstance(idx, str):
            idx = [index for index, item in enumerate(self) if item.name == idx]
            noi = len(idx)
            if noi == 0:
                raise IndexError('Given index does not exist')
            elif noi == 1:
                idx = idx[0]
            else:
                return self.__class__(getattr(self, 'name'), *[self[i] for i in idx])
        elif not isinstance(idx, int) or isinstance(idx, bool):
            if isinstance(idx, bool):
                raise TypeError('Boolean indexing is not supported at the moment')
            try:
                if idx > len(self):
                    raise IndexError(f'Given index {idx} is out of bounds')
            except TypeError:
                raise IndexError('Index must be of type `int`/`slice` or an item name (`str`)')
        keys = list(self._kwargs.keys())
        return self._kwargs[keys[idx]]

    def __setitem__(self, key: int, value: Union[B, V]) -> None:
        """
        Set an item via it's index.

        :param key: Index in self.
        :type key: int
        :param value: Value which index key should be set to.
        :type value: Any
        """
        if isinstance(value, Number):  # noqa: S3827
            item = self.__getitem__(key)
            item.value = value
        elif issubclass(type(value), (BasedBase, DescriptorBase)):
            update_key = list(self._kwargs.keys())
            values = list(self._kwargs.values())
            old_item = values[key]
            # Update the internal dict
            update_dict = {update_key[key]: value}
            self._kwargs.update(update_dict)
            # ADD EDGE
            self._global_object.map.add_edge(self, value)
            self._global_object.map.reset_type(value, 'created_internal')
            value.interface = self.interface
            # REMOVE EDGE
            self._global_object.map.prune_vertex_from_edge(self, old_item)
        else:
            raise NotImplementedError('At the moment only numerical values or EasyScience objects can be set.')

    def __delitem__(self, key: int) -> None:
        """
        Try to delete  an idem by key.

        :param key:
        :type key:
        :return:
        :rtype:
        """
        keys = list(self._kwargs.keys())
        item = self._kwargs[keys[key]]
        self._global_object.map.prune_vertex_from_edge(self, item)
        del self._kwargs[keys[key]]

    def __len__(self) -> int:
        """
        Get the number of items in this collection

        :return: Number of items in this collection.
        :rtype: int
        """
        return len(self._kwargs.keys())

    def _convert_to_dict(self, in_dict, encoder, skip: List[str] = [], **kwargs) -> dict:
        """
        Convert ones self into a serialized form.

        :return: dictionary of ones self
        :rtype: dict
        """
        d = {}
        if hasattr(self, '_modify_dict'):
            # any extra keys defined on the inheriting class
            d = self._modify_dict(skip=skip, **kwargs)
        in_dict['data'] = [encoder._convert_to_dict(item, skip=skip, **kwargs) for item in self]
        out_dict = {**in_dict, **d}
        return out_dict

    @property
    def data(self) -> Tuple:
        """
        The data function returns a tuple of the keyword arguments passed to the
        constructor. This is useful for when you need to pass in a dictionary of data
        to other functions, such as with matplotlib's plot function.

        :param self: Access attributes of the class within the method
        :return: The values of the attributes in a tuple
        :doc-author: Trelent
        """
        return tuple(self._kwargs.values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} `{getattr(self, 'name')}` of length {len(self)}"

    def sort(self, mapping: Callable[[Union[B, V]], Any], reverse: bool = False) -> None:
        """
        Sort the collection according to the given mapping.

        :param mapping: mapping function to sort the collection. i.e. lambda parameter: parameter.value
        :type mapping: Callable
        :param reverse: Reverse the sorting.
        :type reverse: bool
        """
        i = list(self._kwargs.items())
        i.sort(key=lambda x: mapping(x[1]), reverse=reverse)
        self._kwargs.reorder(**{k[0]: k[1] for k in i})
