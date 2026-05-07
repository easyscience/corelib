# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any
from typing import Optional

from easyscience.global_object.undo_redo import property_stack

from .descriptor_base import DescriptorBase


class DescriptorBool(DescriptorBase):
    """A ``Descriptor`` for boolean values."""

    def __init__(
        self,
        name: str,
        value: bool,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        parent: Optional[Any] = None,
    ):
        if not isinstance(value, bool):
            raise ValueError(f'{value=} must be type bool')
        super().__init__(
            name=name,
            unique_name=unique_name,
            description=description,
            url=url,
            display_name=display_name,
            parent=parent,
        )
        if not isinstance(value, bool):
            raise TypeError(f'{value=} must be type bool')
        self._bool_value = value

    @property
    def value(self) -> bool:
        """
        Get the value of self.

        Returns
        -------
        bool
            Value of self.
        """
        return self._bool_value

    @value.setter
    @property_stack
    def value(self, value: bool) -> None:
        """
        Set the value of self.

        Parameters
        ----------
        value : bool
            New value of self.

        Returns
        -------
        None
            None.

        Raises
        ------
        TypeError
            If ``value`` is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError(f'{value=} must be type bool')
        self._bool_value = value

    def __repr__(self) -> str:
        """Return printable representation."""
        class_name = self.__class__.__name__
        obj_name = self._name
        obj_value = self._bool_value
        return f"<{class_name} '{obj_name}': {obj_value}>"

    # To get return type right
    def __copy__(self) -> DescriptorBool:
        """Copy function."""
        return super().__copy__()
