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


class DescriptorContainer(DescriptorBase):
    """
    A `Descriptor` for Array values with units.  The internal representation is a scipp array.
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
        """Constructor for the DescriptorContainer class

        param name: Name of the descriptor
        param value: Value of the descriptor
        param unit: Unit of the descriptor
        param variance: Variance of the descriptor
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
        Get the value. This should be usable for most cases. The full value can be obtained from `obj.full_value`.

        :return: Value of self with unit.
        """
        return self._value

    @value.setter
    @property_stack_deco
    def value(self, value: Union[list, np.ndarray]) -> None:
        """
        Set the value of self. Ensures the input is an array and matches the shape of the existing array.
        The full value can be obtained from `obj.full_value`.

        :param value: New value for the DescriptorContainer, must be a list or numpy array.
        """
       

        self._value = value



    # Just to get return type right
    def __copy__(self) -> DescriptorContainer:
        return super().__copy__()

    def __repr__(self) -> str:
        """
        Return a string representation of the DescriptorContainer, showing its name, value, variance, and unit.
        Large arrays are summarized for brevity.
        """
        # Base string with name
        string = f"<{self.__class__.__name__} '{self._name}': "

        # # Summarize array values
        # values_summary = np.array2string(
        #     self._array.values, 
        #     precision=4, 
        #     threshold=10,  # Show full array if <=10 elements, else summarize
        #     edgeitems=3,   # Show first and last 3 elements for large arrays
        # )
        # string += f"values={values_summary}"

        # # Add errors if they exists
        # if self._array.variances is not None:
        #     errors_summary = np.array2string(
        #         self.error, 
        #         precision=4, 
        #         threshold=10, 
        #         edgeitems=3,
        #     )
        #     string += f", errors={errors_summary}"

        # # Add unit
        # obj_unit = str(self._array.unit)
        # if obj_unit and obj_unit != "dimensionless":
        #     string += f", unit={obj_unit}"

        string += ">"
        return string    

    def as_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        raw_dict = super().as_dict(skip=skip)
        raw_dict['value'] = self._value
        return raw_dict

