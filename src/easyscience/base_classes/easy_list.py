# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import warnings
from collections.abc import MutableSequence
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import overload

from easyscience.io.serializer_base import SerializerBase
from easyscience.variable.descriptor_base import DescriptorBase

from .model_base import ModelBase
from .new_base import NewBase

ProtectedType_ = TypeVar('ProtectedType', bound=NewBase)


class EasyList(ModelBase, MutableSequence[ProtectedType_]):
    # If we were to inherit from List instead of MutableSequence,
    # we would have to overwrite "extend", "remove", "__iadd__", "count", "append", "__iter__" and "clear"
    def __init__(
        self,
        *args: ProtectedType_ | list[ProtectedType_],
        protected_types: list[Type[NewBase]] | Type[NewBase] | None = None,
        unique_name: Optional[str] = None,
        display_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the EasyList.

        :param args: Initial items to add to the list
        :param protected_types: Types that are allowed in the list. Can
            be a single NewBase subclass or a list of them. If None,
            defaults to [NewBase].
        :param unique_name: Optional unique name for the list
        :param display_name: Optional display name for the list
        """
        super().__init__(unique_name=unique_name, display_name=display_name)
        if protected_types is None:
            self._protected_types = [NewBase]
        elif isinstance(protected_types, type) and issubclass(protected_types, NewBase):
            self._protected_types = [protected_types]
        elif isinstance(protected_types, Iterable) and all(
            issubclass(t, NewBase) for t in protected_types
        ):
            self._protected_types = list(protected_types)
        else:
            raise TypeError(
                'protected_types must be a NewBase subclass or an iterable of NewBase subclasses'
            )
        self._data: List[ProtectedType_] = []

        # Add initial items
        for item in args:
            if isinstance(item, list):
                for sub_item in item:
                    self.append(sub_item)
            else:
                self.append(item)

        # For deserialization, the dict can't contain an *args, so we check for 'data' in kwargs
        if 'data' in kwargs:
            data = kwargs.pop('data')
            for item in data:
                self.append(item)

    # MutableSequence abstract methods

    # Use @overload to provide precise type hints for different __getitem__ argument types
    @overload
    def __getitem__(self, idx: int) -> ProtectedType_: ...
    @overload
    def __getitem__(self, idx: slice) -> 'EasyList[ProtectedType_]': ...
    @overload
    def __getitem__(self, idx: str) -> ProtectedType_: ...
    def __getitem__(self, idx: int | slice | str) -> ProtectedType_ | 'EasyList[ProtectedType_]':
        """Get an item by index, slice, or unique_name.

        :param idx: Index, slice, or unique_name of the item
        :return: The item or a new EasyList for slices
        """
        if isinstance(idx, int):
            return self._data[idx]
        elif isinstance(idx, slice):
            return self.__class__(self._data[idx], protected_types=self._protected_types)
        elif isinstance(idx, str):
            element = next((r for r in self._data if self._get_key(r) == idx), None)
            if element is not None:
                return element
            raise KeyError(f'No item with unique name "{idx}" found')
        else:
            raise TypeError('Index must be an int, slice, or str')

    @overload
    def __setitem__(self, idx: int, value: ProtectedType_) -> None: ...
    @overload
    def __setitem__(self, idx: slice, value: Iterable[ProtectedType_]) -> None: ...

    def __setitem__(
        self, idx: int | slice, value: ProtectedType_ | Iterable[ProtectedType_]
    ) -> None:
        """Set an item at an index.

        :param idx: Index to set
        :param value: New value
        """
        if isinstance(idx, int):
            if not isinstance(value, tuple(self._protected_types)):
                raise TypeError(f'Items must be one of {self._protected_types}, got {type(value)}')
            if value is not self._data[idx] and value in self:
                warnings.warn(
                    f'Item with unique name "{self._get_key(value)}" already in EasyList, it will be ignored'
                )
                return
            self._data[idx] = value
        elif isinstance(idx, slice):
            if not isinstance(value, Iterable):
                raise TypeError('Value must be an iterable for slice assignment')
            replaced = self._data[idx]
            new_values = list(value)
            if len(new_values) != len(replaced):
                raise ValueError(
                    'Length of new values must match the length of the slice being replaced'
                )
            for i, v in enumerate(new_values):
                if not isinstance(v, tuple(self._protected_types)):
                    raise TypeError(f'Items must be one of {self._protected_types}, got {type(v)}')
                if v in self and replaced[i] is not v:
                    warnings.warn(
                        f'Item with unique name "{v.unique_name}" already in EasyList, it will be ignored'
                    )
                    new_values[i] = replaced[
                        i
                    ]  # Keep the original value if the new one is a duplicate
            self._data[idx] = new_values
        else:
            raise TypeError('Index must be an int or slice')

    def __delitem__(self, idx: int | slice | str) -> None:
        """Delete an item by index, slice, or name.

        :param idx: Index, slice, or name of item to delete
        """
        if isinstance(idx, (int, slice)):
            del self._data[idx]
        elif isinstance(idx, str):
            for i, item in enumerate(self._data):
                if self._get_key(item) == idx:
                    del self._data[i]
                    return
            raise KeyError(f'No item with unique name "{idx}" found')
        else:
            raise TypeError('Index must be an int, slice, or str')

    def __len__(self) -> int:
        """Return the number of items in the collection."""
        return len(self._data)

    def insert(self, index: int, value: ProtectedType_) -> None:
        """Insert an item at an index.

        :param index: Index to insert at
        :param value: Item to insert
        """
        if not isinstance(index, int):
            raise TypeError('Index must be an integer')
        elif not isinstance(value, tuple(self._protected_types)):
            raise TypeError(f'Items must be one of {self._protected_types}, got {type(value)}')
        if value in self:
            warnings.warn(
                f'Item with unique name "{self._get_key(value)}" already in EasyList, it will be ignored'
            )
            return
        self._data.insert(index, value)

    def _get_key(self, obj) -> str:
        """Get the unique name of an object.

        Can be overridden to use a different attribute as the key.
        :param object: Object to get the key for
        :return: The key of the object
        :rtype: str
        """
        return obj.unique_name

    def get_all_variables(self) -> List[DescriptorBase]:
        """Get all `Descriptor` and `Parameter` objects from all
        elements that are derived from `ModelBase`.

        For each element that is a `ModelBase` instance, the element's
        own `get_all_variables()` method is called and the results are
        collected into a single flat list.

        :return: Flat list of all `DescriptorBase` objects from all
            `ModelBase` elements.
        :rtype: List[DescriptorBase]
        """
        all_vars: List[DescriptorBase] = []
        for item in self._data:
            if isinstance(item, ModelBase):
                all_vars.extend(item.get_all_variables())
        return all_vars

    # Overwriting methods

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__} of length {len(self)} of type(s) {self._protected_types}'
        )

    def __contains__(self, item: ProtectedType_ | str) -> bool:
        if isinstance(item, str):
            return any(self._get_key(r) == item for r in self._data)
        return item in self._data

    def __reversed__(self):
        return self._data.__reversed__()

    def sort(self, key: Callable[[ProtectedType_], Any] = None, reverse: bool = False) -> None:
        """Sort the collection according to the given key function.

        :param key: Mapping function to sort by
        :param reverse: Whether to reverse the sort
        """
        self._data.sort(reverse=reverse, key=key)

    def index(self, value: ProtectedType_ | str, start: int = 0, stop: int = None) -> int:
        if stop is None:
            stop = len(self._data)
        if isinstance(value, str):
            for i in range(start, min(stop, len(self._data))):
                if self._get_key(self._data[i]) == value:
                    return i
            raise ValueError(f'{value} is not in EasyList')
        return self._data.index(value, start, stop)

    def pop(self, index: int | str = -1) -> ProtectedType_:
        """Remove and return an item at the given index or unique_name.

        :param index: Index or unique_name of the item to remove
        :return: The removed item
        """
        if isinstance(index, int):
            return self._data.pop(index)
        elif isinstance(index, str):
            for i, item in enumerate(self._data):
                if self._get_key(item) == index:
                    return self._data.pop(i)
            raise KeyError(f'No item with unique name "{index}" found')
        else:
            raise TypeError('Index must be an int or str')

    # Serialization support

    def to_dict(self) -> dict:
        """Convert the EasyList to a dictionary for serialization.

        :return: Dictionary representation of the EasyList
        """
        dict_repr = super().to_dict()
        if self._protected_types != [NewBase]:
            dict_repr['protected_types'] = [
                {'@module': cls_.__module__, '@class': cls_.__name__}
                for cls_ in self._protected_types
            ]  # noqa: E501
        dict_repr['data'] = [item.to_dict() for item in self._data]
        return dict_repr

    @classmethod
    def from_dict(cls, obj_dict: Dict[str, Any]) -> NewBase:
        """Re-create an EasyScience object from a full encoded
        dictionary.

        :param obj_dict: dictionary containing the serialized contents (from `SerializerDict`) of an EasyScience object
        :return: Reformed EasyScience object
        """
        if not SerializerBase._is_serialized_easyscience_object(obj_dict):
            raise ValueError(
                'Input must be a dictionary representing an EasyScience EasyList object.'
            )
        temp_dict = copy.deepcopy(obj_dict)  # Make a copy to avoid mutating the input
        if temp_dict['@class'] == cls.__name__:
            if 'protected_types' in temp_dict:
                protected_types = temp_dict.pop('protected_types')
                for i, type_dict in enumerate(protected_types):
                    if '@module' in type_dict and '@class' in type_dict:
                        modname = type_dict['@module']
                        classname = type_dict['@class']
                        mod = __import__(modname, globals(), locals(), [classname], 0)
                        if hasattr(mod, classname):
                            cls_ = getattr(mod, classname)
                            protected_types[i] = cls_
                        else:
                            raise ImportError(
                                f'Could not import class {classname} from module {modname}'
                            )
                    else:
                        raise ValueError(
                            'Each protected type must be a serialized EasyScience class with @module and @class keys'
                        )  # noqa: E501
            else:
                protected_types = None
            kwargs = SerializerBase.deserialize_dict(temp_dict)
            data = kwargs.pop('data', [])
            return cls(data, protected_types=protected_types, **kwargs)
        else:
            raise ValueError(
                f'Class name in dictionary does not match the expected class: {cls.__name__}.'
            )
