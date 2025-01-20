from __future__ import annotations

from typing import Any
from typing import Optional

from easyscience.global_object.undo_redo import property_stack_deco

from .descriptor_base import DescriptorBase


class DescriptorStr(DescriptorBase):
    """
    A `Descriptor` for string values.
    """

    def __init__(
        self,
        name: str,
        value: str,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        parent: Optional[Any] = None,
    ):
        super().__init__(
            name=name,
            unique_name=unique_name,
            description=description,
            url=url,
            display_name=display_name,
            parent=parent,
        )
        if not isinstance(value, str):
            raise ValueError(f'{value=} must be type str')
        self._string = value

    @property
    def value(self) -> str:
        """
        Get the value of self.

        :return: Value of self with unit.
        """
        return self._string

    @value.setter
    @property_stack_deco
    def value(self, value: str) -> None:
        """
        Set the value of self.

        :param value: New value of self
        :return: None
        """
        if not isinstance(value, str):
            raise ValueError(f'{value=} must be type str')
        self._string = value

    def __repr__(self) -> str:
        """Return printable representation."""
        class_name = self.__class__.__name__
        obj_name = self._name
        obj_value = self._string
        return f"<{class_name} '{obj_name}': {obj_value}>"

    # To get return type right
    def __copy__(self) -> DescriptorStr:
        return super().__copy__()
