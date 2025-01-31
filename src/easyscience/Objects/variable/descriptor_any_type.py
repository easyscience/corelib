from __future__ import annotations

import numbers
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from easyscience.global_object.undo_redo import property_stack_deco

from .descriptor_base import DescriptorBase


class DescriptorAnyType(DescriptorBase):
    """
    A `Descriptor` for any type that does not fit the other Descriptors. Should be avoided when possible. 
    It was created to hold the symmetry operations used in the SpaceGroup class of EasyCrystallography.
    """

    def __init__(
        self,
        name: str,
        value: Any,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        parent: Optional[Any] = None,
    ):
        """Constructor for the DescriptorAnyType class

        param name: Name of the descriptor
        param value: Value of the descriptor
        param description: Description of the descriptor
        param url: URL of the descriptor
        param display_name: Display name of the descriptor
        param parent: Parent of the descriptor
        .. note:: Undo/Redo functionality is implemented for the attributes `variance`, `error`, `unit` and `value`.
        """

        self._value=value
        
        super().__init__(
            name=name,
            unique_name=unique_name,
            description=description,
            url=url,
            display_name=display_name,
            parent=parent,
        )

    @property
    def value(self) -> numbers.Number:
        """
        Get the value. 

        :return: Value of self.
        """
        return self._value

    @value.setter
    @property_stack_deco
    def value(self, value: Union[list, np.ndarray]) -> None:
        """
        Set the value of self. 

        :param value: New value for the DescriptorAnyType.
        """
        self._value = value

    def __copy__(self) -> DescriptorAnyType:
        return super().__copy__()

    def __repr__(self) -> str:
        """
        Return a string representation of the DescriptorAnyType, showing its name and value.
        """

        if hasattr(self._value, '__repr__'):
            value_repr = repr(self._value)
        else:
            value_repr = type(self._value)

        return f"<{self.__class__.__name__} '{self._name}': {value_repr}>"

    def as_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        raw_dict = super().as_dict(skip=skip)
        raw_dict['value'] = self._value
        return raw_dict

