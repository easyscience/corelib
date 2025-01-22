from __future__ import annotations

import numbers
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import scipp as sc
from scipp import UnitError
from scipp import Variable

from easyscience.global_object.undo_redo import PropertyStack
from easyscience.global_object.undo_redo import property_stack_deco

from .descriptor_base import DescriptorBase


class DescriptorArray(DescriptorBase):
    """
    A `Descriptor` for Array values with units.  The internal representation is a scipp array.
    """

    def __init__(
        self,
        name: str,
        value: numbers.Number,
        unit: Optional[Union[str, sc.Unit]] = '',
        variance: Optional[numbers.Number] = None,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        parent: Optional[Any] = None,
    ):
        """Constructor for the DescriptorArray class

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

        if not isinstance(value, (list, np.ndarray)):
            raise TypeError(f"{value=} must be a list or numpy array.")
        if isinstance(value, list):
            value = np.array(value)  # Convert to numpy array for consistent handling.

        if variance is not None:
            if not isinstance(variance, (list, np.ndarray)):
                raise TypeError(f"{variance=} must be a list or numpy array if provided.")
            if isinstance(variance, list):
                variance = np.array(variance)  # Convert to numpy array for consistent handling.
            if variance.shape != value.shape:
                raise ValueError(f"{variance=} must have the same shape as {value=}.")
            if not np.all(variance >= 0):
                raise ValueError(f"{variance=} must only contain non-negative values.")
            
        if not isinstance(unit, sc.Unit) and not isinstance(unit, str):
            raise TypeError(f'{unit=} must be a scipp unit or a string representing a valid scipp unit')

        try:
            self._array = sc.array(dims=['row', 'column'], values=value, unit=unit, variances=variance)
        except Exception as message:
            raise UnitError(message)
                
        self._array = sc.array(dims=['row','column'],values=value, unit=unit, variances=variance)
        
        super().__init__(
            name=name,
            unique_name=unique_name,
            description=description,
            url=url,
            display_name=display_name,
            parent=parent,
        )

        # Call convert_unit during initialization to ensure that the unit has no numbers in it, and to ensure unit consistency.
        if self.unit is not None:
            self.convert_unit(self._base_unit())

    @classmethod
    def from_scipp(cls, name: str, full_value: Variable, **kwargs) -> DescriptorArray:
        """
        Create a DescriptorArray from a scipp array.

        :param name: Name of the descriptor
        :param full_value: Value of the descriptor as a scipp variable
        :param kwargs: Additional parameters for the descriptor
        :return: DescriptorArray
        """
        if not isinstance(full_value, Variable):
            raise TypeError(f'{full_value=} must be a scipp array')
        if len(full_value.dims) != 0:
            raise TypeError(f'{full_value=} must be a scipp array')
        return cls(name=name, value=full_value.values, unit=full_value.unit, variance=full_value.variances, **kwargs)

    @property
    def full_value(self) -> Variable:
        """
        Get the value of self as a scipp array. This is should be usable for most cases.

        :return: Value of self with unit.
        """
        return self._array

    @full_value.setter
    def full_value(self, full_value: Variable) -> None:
        raise AttributeError(
            f'Full_value is read-only. Change the value and variance seperately. Or create a new {self.__class__.__name__}.'
        )

    @property
    def value(self) -> numbers.Number:
        """
        Get the value. This should be usable for most cases. The full value can be obtained from `obj.full_value`.

        :return: Value of self with unit.
        """
        return self._array.values

    @value.setter
    @property_stack_deco
    def value(self, value: Union[list, np.ndarray]) -> None:
        """
        Set the value of self. Ensures the input is an array and matches the shape of the existing array.
        The full value can be obtained from `obj.full_value`.

        :param value: New value for the DescriptorArray, must be a list or numpy array.
        """
        if not isinstance(value, (list, np.ndarray)):
            raise TypeError(f"{value=} must be a list or numpy array.")
        if isinstance(value, list):
            value = np.array(value)  # Convert lists to numpy arrays for consistent handling.

        if value.shape != self._array.values.shape:
            raise ValueError(f"{value=} must have the same shape as the existing array values.")

        self._array.values = value

    @property
    def unit(self) -> str:
        """
        Get the unit.

        :return: Unit as a string.
        """
        return str(self._array.unit)

    @unit.setter
    def unit(self, unit_str: str) -> None:
        raise AttributeError(
            (
                f'Unit is read-only. Use convert_unit to change the unit between allowed types '
                f'or create a new {self.__class__.__name__} with the desired unit.'
            )
        )  # noqa: E501

    @property
    def variance(self) -> float:
        """
        Get the variance.

        :return: variance.
        """
        return self._array.variance

    @variance.setter
    @property_stack_deco
    def variance(self, variance: Union[list, np.ndarray]) -> None:
        """
        Set the variance of self. Ensures the input is an array and matches the shape of the existing values.

        :param variance: New variance for the DescriptorArray, must be a list or numpy array.
        """
        if variance is not None:
            if not isinstance(variance, (list, np.ndarray)):
                raise TypeError(f"{variance=} must be a list or numpy array.")
            if isinstance(variance, list):
                variance = np.array(variance)  # Convert lists to numpy arrays for consistent handling.

            if variance.shape != self._array.values.shape:
                raise ValueError(f"{variance=} must have the same shape as the array values.")

            if not np.all(variance >= 0):
                raise ValueError(f"{variance=} must only contain non-negative values.")

        self._array.variances = variance
        
    @property
    def error(self) -> Optional[np.ndarray]:
        """
        The standard deviations , calculated as the square root of variances.

        :return: A numpy array of standard deviations, or None if variances are not set.
        """
        if self._array.variances is None:
            return None
        return np.sqrt(self._array.variances)

    @error.setter
    @property_stack_deco
    def error(self, error: Union[list, np.ndarray]) -> None:
        """
        Set the standard deviation for the parameter, which updates the variances.

        :param error: A list or numpy array of standard deviations.
        """
        if error is not None:
            if not isinstance(error, (list, np.ndarray)):
                raise TypeError(f"{error=} must be a list or numpy array.")
            if isinstance(error, list):
                error = np.array(error)  # Convert lists to numpy arrays for consistent handling.

            if error.shape != self._array.values.shape:
                raise ValueError(f"{error=} must have the same shape as the array values.")

            if not np.all(error >= 0):
                raise ValueError(f"{error=} must only contain non-negative values.")

            # Update variances as the square of the errors
            self._array.variances = error**2
        else:
            self._array.variances = None
            

    def convert_unit(self, unit_str: str) -> None:
        """
        Convert the value from one unit system to another.

        :param unit_str: New unit in string form
        """
        if not isinstance(unit_str, str):
            raise TypeError(f'{unit_str=} must be a string representing a valid scipp unit')
        try:
            new_unit = sc.Unit(unit_str)
        except UnitError as message:
            raise UnitError(message) from None

        # Save the current state for undo/redo
        old_array = self._array

        # Perform the unit conversion
        try:
            new_array = self._array.to(unit=new_unit)
        except Exception as e:
            raise UnitError(f"Failed to convert unit: {e}") from e

        # Define the setter function for the undo stack
        def set_array(obj, scalar):
            obj._array = scalar

        # Push to undo stack
        self._global_object.stack.push(
            PropertyStack(self, set_array, old_array, new_array, text=f"Convert unit to {unit_str}")
        )

        # Update the array
        self._array = new_array


    # Just to get return type right
    def __copy__(self) -> DescriptorArray:
        return super().__copy__()

    def __repr__(self) -> str:
        """
        Return a string representation of the DescriptorArray, showing its name, value, variance, and unit.
        Large arrays are summarized for brevity.
        """
        # Base string with name
        string = f"<{self.__class__.__name__} '{self._name}': "

        # Summarize array values
        values_summary = np.array2string(
            self._array.values, 
            precision=4, 
            threshold=10,  # Show full array if <=10 elements, else summarize
            edgeitems=3,   # Show first and last 3 elements for large arrays
        )
        string += f"values={values_summary}"

        # Add errors if they exists
        if self._array.variances is not None:
            errors_summary = np.array2string(
                self.error, 
                precision=4, 
                threshold=10, 
                edgeitems=3,
            )
            string += f", errors={errors_summary}"

        # Add unit
        obj_unit = str(self._array.unit)
        if obj_unit and obj_unit != "dimensionless":
            string += f", unit={obj_unit}"

        string += ">"
        return string    

    def as_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        raw_dict = super().as_dict(skip=skip)
        raw_dict['value'] = self._array.values
        raw_dict['unit'] = str(self._array.unit)
        raw_dict['variance'] = self._array.variances
        return raw_dict


    def __add__(self, other: Union[DescriptorArray, list, np.ndarray, numbers.Number]) -> DescriptorArray:
        """
        Perform element-wise addition with another DescriptorArray, numpy array, list, or number.

        :param other: The object to add. Must be a DescriptorArray with compatible units,
                    or a numpy array/list with the same shape if the DescriptorArray is dimensionless.
        :return: A new DescriptorArray representing the result of the addition.
        """
        if isinstance(other, numbers.Number):
            if self.unit not in [None, "dimensionless"]:
                raise UnitError("Numbers can only be added to dimensionless values")
            new_full_value = self.full_value + other # scipp can handle addition with numbers

        elif isinstance(other, (list, np.ndarray)):
            if self.unit not in [None, "dimensionless"]:
                raise UnitError("Addition with numpy arrays or lists is only allowed for dimensionless values")
            
            # Convert `other` to numpy array if it's a list
            if isinstance(other, list):
                other = np.array(other)

            # Ensure dimensions match
            if other.shape != self._array.values.shape:
                raise ValueError(f"Shape of {other=} must match the shape of DescriptorArray values")

            new_value = self._array.values + other
            new_full_value=sc.array(dims=['row','column'],values=new_value,unit=self.unit,variances=self._array.variances)

        elif isinstance(other, DescriptorArray):
            original_unit = other.unit
            try:
                other.convert_unit(self.unit)
            except UnitError:
                raise UnitError(f"Values with units {self.unit} and {other.unit} cannot be added") from None

            # Ensure dimensions match
            if self.full_value.dims != other.full_value.dims:
                raise ValueError(f"Dimensions of the DescriptorArrays do not match: "
                                f"{self.full_value.dims} vs {other.full_value.dims}")

            new_full_value = self.full_value + other.full_value

            # Revert `other` to its original unit
            other.convert_unit(original_unit)
        else:
            return NotImplemented
        
        descriptor_array = DescriptorArray.from_scipp(name=self.name, full_value=new_full_value)
        descriptor_array.name = descriptor_array.unique_name
        return descriptor_array


    def __radd__(self, other: Union[list, np.ndarray, numbers.Number]) -> DescriptorArray:
        """
        Handle reverse addition for numbers, numpy arrays, and lists. Element-wise operation.
        Converts the unit of `self` to match `other` if `other` is a DescriptorArray.

        :param other: The object to add. Must be a DescriptorArray, numpy array, list, or number.
        :return: A new DescriptorArray representing the result of the addition.
        """
        if isinstance(other, DescriptorArray):
            # Ensure the reverse operation respects unit compatibility
            return other.__add__(self)
        else:
            # Delegate to `__add__` for other types
            return self.__add__(other)




# TODO: add arithmetic operations
# They should be allowed between DescriptorArray and numbers, and between DescriptorArray and DescriptorArray.
# The result should be a new DescriptorArray with the same unit as the first argument.

        

    # def __add__(self, other: Union[DescriptorArray, numbers.Number]) -> DescriptorArray: 
    #     if isinstance(other, numbers.Number):
    #         if self.unit != 'dimensionless':
    #             raise UnitError('Numbers can only be added to dimensionless values')
    #         new_value = self.full_value + other
    #     elif type(other) is DescriptorArray:
    #         original_unit = other.unit
    #         try:
    #             other.convert_unit(self.unit)
    #         except UnitError:
    #             raise UnitError(f'Values with units {self.unit} and {other.unit} cannot be added') from None
    #         new_value = self.full_value + other.full_value
    #         other.convert_unit(original_unit)
    #     else:
    #         return NotImplemented
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __radd__(self, other: numbers.Number) -> DescriptorArray:
    #     if isinstance(other, numbers.Number):
    #         if self.unit != 'dimensionless':
    #             raise UnitError('Numbers can only be added to dimensionless values')
    #         new_value = other + self.full_value
    #     else:
    #         return NotImplemented
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __sub__(self, other: Union[DescriptorArray, numbers.Number]) -> DescriptorArray:
    #     if isinstance(other, numbers.Number):
    #         if self.unit != 'dimensionless':
    #             raise UnitError('Numbers can only be subtracted from dimensionless values')
    #         new_value = self.full_value - other
    #     elif type(other) is DescriptorArray:
    #         original_unit = other.unit
    #         try:
    #             other.convert_unit(self.unit)
    #         except UnitError:
    #             raise UnitError(f'Values with units {self.unit} and {other.unit} cannot be subtracted') from None
    #         new_value = self.full_value - other.full_value
    #         other.convert_unit(original_unit)
    #     else:
    #         return NotImplemented
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __rsub__(self, other: numbers.Number) -> DescriptorArray:
    #     if isinstance(other, numbers.Number):
    #         if self.unit != 'dimensionless':
    #             raise UnitError('Numbers can only be subtracted from dimensionless values')
    #         new_value = other - self.full_value
    #     else:
    #         return NotImplemented
    #     descriptor = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor.name = descriptor.unique_name
    #     return descriptor

    # def __mul__(self, other: Union[DescriptorArray, numbers.Number]) -> DescriptorArray:
    #     if isinstance(other, numbers.Number):
    #         new_value = self.full_value * other
    #     elif type(other) is DescriptorArray:
    #         new_value = self.full_value * other.full_value
    #     else:
    #         return NotImplemented
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.convert_unit(descriptor_number._base_unit())
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __rmul__(self, other: numbers.Number) -> DescriptorArray:
    #     if isinstance(other, numbers.Number):
    #         new_value = other * self.full_value
    #     else:
    #         return NotImplemented
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __truediv__(self, other: Union[DescriptorArray, numbers.Number]) -> DescriptorArray:
    #     if isinstance(other, numbers.Number):
    #         original_other = other
    #         if other == 0:
    #             raise ZeroDivisionError('Cannot divide by zero')
    #         new_value = self.full_value / other
    #     elif type(other) is DescriptorArray:
    #         original_other = other.value
    #         if original_other == 0:
    #             raise ZeroDivisionError('Cannot divide by zero')
    #         new_value = self.full_value / other.full_value
    #         other.value = original_other
    #     else:
    #         return NotImplemented
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.convert_unit(descriptor_number._base_unit())
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __rtruediv__(self, other: numbers.Number) -> DescriptorArray:
    #     if isinstance(other, numbers.Number):
    #         if self.value == 0:
    #             raise ZeroDivisionError('Cannot divide by zero')
    #         new_value = other / self.full_value
    #     else:
    #         return NotImplemented
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __pow__(self, other: Union[DescriptorArray, numbers.Number]) -> DescriptorArray:
    #     if isinstance(other, numbers.Number):
    #         exponent = other
    #     elif type(other) is DescriptorArray:
    #         if other.unit != 'dimensionless':
    #             raise UnitError('Exponents must be dimensionless')
    #         if other.variance is not None:
    #             raise ValueError('Exponents must not have variance')
    #         exponent = other.value
    #     else:
    #         return NotImplemented
    #     try:
    #         new_value = self.full_value**exponent
    #     except Exception as message:
    #         raise message from None
    #     if np.isnan(new_value.value):
    #         raise ValueError('The result of the exponentiation is not a number')
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __rpow__(self, other: numbers.Number) -> numbers.Number:
    #     if isinstance(other, numbers.Number):
    #         if self.unit != 'dimensionless':
    #             raise UnitError('Exponents must be dimensionless')
    #         if self.variance is not None:
    #             raise ValueError('Exponents must not have variance')
    #         new_value = other**self.value
    #     else:
    #         return NotImplemented
    #     return new_value

    # def __neg__(self) -> DescriptorArray:
    #     new_value = -self.full_value
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    # def __abs__(self) -> DescriptorArray:
    #     new_value = abs(self.full_value)
    #     descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
    #     descriptor_number.name = descriptor_number.unique_name
    #     return descriptor_number

    def _base_unit(self) -> str:
        string = str(self._array.unit)
        for i, letter in enumerate(string):
            if letter == 'e':
                if string[i : i + 2] not in ['e+', 'e-']:
                    return string[i:]
            elif letter not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '+', '-']:
                return string[i:]
        return ''



    # TODO: add matrix multiplication and division using numpy.
