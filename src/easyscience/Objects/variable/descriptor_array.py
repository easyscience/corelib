from __future__ import annotations

import numbers
import operator as op
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from warnings import warn

import numpy as np
import scipp as sc
from scipp import UnitError
from scipp import Variable

from easyscience.global_object.undo_redo import PropertyStack
from easyscience.global_object.undo_redo import property_stack_deco

from .descriptor_base import DescriptorBase
from .descriptor_number import DescriptorNumber


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
                # TODO: handle 1xn and nx1 arrays
        self._array = sc.array(dims=['row','column'], values=value, unit=unit, variances=variance)
        
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
        return self._array.variances

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
        The standard deviations, calculated as the square root of variances.

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
        string=string.replace('\n', ',')
        return string    


    def as_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        raw_dict = super().as_dict(skip=skip)
        raw_dict['value'] = self._array.values
        raw_dict['unit'] = str(self._array.unit)
        raw_dict['variance'] = self._array.variances
        return raw_dict
    
    def _smooth_operator(self,
                         other: Union[DescriptorArray, DescriptorNumber, list, numbers.Number],
                         operator: str,
                         units_must_match: bool = True) -> DescriptorArray:
        """
        Perform element-wise operations with another DescriptorNumber, DescriptorArray, list, or number.

        :param other: The object to operate on. Must be a DescriptorArray or DescriptorNumber with compatible units,
                    or a list with the same shape if the DescriptorArray is dimensionless.
        :param operator: The operation to perform
        :return: A new DescriptorArray representing the result of the operation.
        """
        if isinstance(other, numbers.Number):
            # Does not need to be dimensionless for multiplication and division
            if self.unit not in [None, "dimensionless"] and units_must_match:
                raise UnitError("Numbers can only be used together with dimensionless values")
            new_full_value = operator(self.full_value, other)

        elif isinstance(other, list):
            if self.unit not in [None, "dimensionless"] and units_must_match:
                raise UnitError("Operations with lists are only allowed for dimensionless values")
            
            # Ensure dimensions match
            if np.shape(other) != self._array.values.shape:
                raise ValueError(f"Shape of {other=} must match the shape of DescriptorArray values")
            
            other = sc.array(dims=self._array.dims, values=other)
            new_full_value = operator(self._array, other)  # Let scipp handle operation for uncertainty propagation
        
        elif isinstance(other, DescriptorNumber):
            try:
                other_converted = other.__copy__()
                other_converted.convert_unit(self.unit)
            except UnitError:
                if units_must_match:
                    raise UnitError(f"Values with units {self.unit} and {other.unit} are not compatible") from None
            # Operations with a DescriptorNumber that has a variance WILL introduce
            # correlations between the elements of the DescriptorArray.
            # See, https://content.iospress.com/articles/journal-of-neutron-research/jnr220049
            # However, DescriptorArray does not consider the covariance between
            # elements of the array. Hence, the broadcasting is "manually"
            # performed to work around `scipp` and a warning raised to the end user.
            if (self._array.variances is not None or other.variance is not None):
                warn('Correlations introduced by this operation will not be considered.\
                      See https://content.iospress.com/articles/journal-of-neutron-research/jnr220049\
                      for further detailes', UserWarning)
            # Cheeky copy() of broadcasted scipp array to force scipp to perform the broadcast here
            broadcasted = sc.broadcast(other_converted.full_value, 
                                             dims=self._array.dims,
                                             shape=self._array.shape).copy()  
            new_full_value = operator(self.full_value, broadcasted)

        elif isinstance(other, DescriptorArray):
            try:
                other_converted = other.__copy__()
                other_converted.convert_unit(self.unit)
            except UnitError:
                if units_must_match:
                    raise UnitError(f"Values with units {self.unit} and {other.unit} are incompatible") from None

            # Ensure dimensions match
            if self.full_value.dims != other_converted.full_value.dims:
                raise ValueError(f"Dimensions of the DescriptorArrays do not match: "
                                f"{self.full_value.dims} vs {other_converted.full_value.dims}")

            new_full_value = operator(self.full_value, other_converted.full_value)

        else:
            return NotImplemented

        descriptor_array = DescriptorArray.from_scipp(name=self.name, full_value=new_full_value)
        descriptor_array.name = descriptor_array.unique_name
        return descriptor_array

    def _rsmooth_operator(self,
                          other: Union[DescriptorArray, DescriptorNumber, list, numbers.Number],
                          operator: str,
                          units_must_match: bool = True) -> DescriptorArray:
        """
        Handle reverse operations for DescriptorArrays, DescriptorNumbers, lists, and scalars.
        Ensures unit compatibility when `other` is a DescriptorNumber.
        """
        def reversed_operator(a, b):
            return operator(b, a)
        if isinstance(other, DescriptorArray):
            # This is probably never called
            return operator(other, self)
        elif isinstance(other, DescriptorNumber):
            # Ensure unit compatibility for DescriptorNumber
            original_unit = self.unit
            try:
                self.convert_unit(other.unit)  # Convert `self` to `other`'s unit
            except UnitError:
                # Only allowed operations with different units are
                # multiplication and division. We try to convert
                # the units for mul/div, but if the conversion
                # fails it's no big deal.
                if units_must_match:
                    raise UnitError(f"Values with units {self.unit} and {other.unit} are incompatible") from None
            result = self._smooth_operator(other, reversed_operator, units_must_match)
            # Revert `self` to its original unit
            self.convert_unit(original_unit)
            return result
        else:
            # Delegate to operation to __self__ for other types (e.g., list, scalar)
            return self._smooth_operator(other, reversed_operator, units_must_match)
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): 
        """
        DescriptorArray does not generally support Numpy array functions.
        For example, `np.argwhere(descriptorArray: DescriptorArray)` should fail.
        Modify this function if you want to add such functionality.
        """
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        """
        DescriptorArray does not generally support Numpy array functions.
        For example, `np.argwhere(descriptorArray: DescriptorArray)` should fail.
        Modify this function if you want to add such functionality.
        """
        return NotImplemented
    
    def __add__(self, other: Union[DescriptorArray, DescriptorNumber, list, numbers.Number]) -> DescriptorArray:
        """
        Perform element-wise addition with another DescriptorNumber, DescriptorArray, list, or number.

        :param other: The object to add. Must be a DescriptorArray or DescriptorNumber with compatible units,
                    or a list with the same shape if the DescriptorArray is dimensionless.
        :return: A new DescriptorArray representing the result of the addition.
        """
        return self._smooth_operator(other, op.add)

    def __radd__(self, other: Union[DescriptorArray, DescriptorNumber, list, numbers.Number]) -> DescriptorArray:
        """
        Handle reverse addition for DescriptorArrays, DescriptorNumbers, lists, and scalars.
        Ensures unit compatibility when `other` is a DescriptorNumber.
        """
        return self._rsmooth_operator(other, op.add)
        
    def __sub__(self, other: Union[DescriptorArray, list, np.ndarray, numbers.Number]) -> DescriptorArray:
        """
        Perform element-wise subtraction with another DescriptorArray, list, or number.

        :param other: The object to subtract. Must be a DescriptorArray with compatible units,
                    or a list with the same shape if the DescriptorArray is dimensionless.
        :return: A new DescriptorArray representing the result of the subtraction.
        """
        if isinstance(other, (DescriptorArray, DescriptorNumber, list, numbers.Number)):
            # Leverage __neg__ and __add__ for subtraction
            if isinstance(other, list):
                # Use numpy to negate all elements of the list
                value = (-np.array(other)).tolist()
            else:
                value = -other
            return self.__add__(value)
        else:
            return NotImplemented
        
    def __rsub__(self, other: Union[DescriptorArray, list, numbers.Number]) -> DescriptorArray:
        """
        Perform element-wise subtraction with another DescriptorArray, list, or number.

        :param other: The object to subtract. Must be a DescriptorArray with compatible units,
                    or a list with the same shape if the DescriptorArray is dimensionless.
        :return: A new DescriptorArray representing the result of the subtraction.
        """
        if isinstance(other, (DescriptorArray, DescriptorNumber, list, numbers.Number)):
            if isinstance(other, list):
                # Use numpy to negate all elements of the list
                value = (-np.array(other)).tolist()
            else:
                value = -other
            return -(self.__radd__(value))
        else:
            return NotImplemented
    
    def __mul__(self, other: Union[DescriptorArray, DescriptorNumber, list, numbers.Number]) -> DescriptorArray:
        """
        Perform element-wise multiplication with another DescriptorNumber, DescriptorArray, list, or number.

        :param other: The object to multiply. Must be a DescriptorArray or DescriptorNumber with compatible units,
                    or a list with the same shape if the DescriptorArray is dimensionless.
        :return: A new DescriptorArray representing the result of the addition.
        """
        if not isinstance(other, (DescriptorArray, DescriptorNumber, list, numbers.Number)):
            return NotImplemented
        return self._smooth_operator(other, op.mul, units_must_match=False)
    
    def __rmul__(self, other: Union[DescriptorArray, DescriptorNumber, list, numbers.Number]) -> DescriptorArray:
        """
        Handle reverse multiplication for DescriptorArrays, DescriptorNumbers, lists, and scalars.
        Ensures unit compatibility when `other` is a DescriptorNumber.
        """
        if not isinstance(other, (DescriptorArray, DescriptorNumber, list, numbers.Number)):
            return NotImplemented
        return self._rsmooth_operator(other, op.mul, units_must_match=False)
    
    def __truediv__(self, other: Union[DescriptorArray, DescriptorNumber, list, numbers.Number]) -> DescriptorArray:
        """
        Perform element-wise division with another DescriptorNumber, DescriptorArray, list, or number.

        :param other: The object to use as a denominator. Must be a DescriptorArray or DescriptorNumber with compatible units,
                    or a list with the same shape if the DescriptorArray is dimensionless.
        :return: A new DescriptorArray representing the result of the addition.
        """
        if not isinstance(other, (DescriptorArray, DescriptorNumber, list, numbers.Number)):
            return NotImplemented

        if isinstance(other, numbers.Number):
            original_other = other
        elif isinstance(other, (numbers.Number, list)):
            original_other = np.array(other)
        elif isinstance(other, DescriptorNumber):
            original_other = other.value
        elif isinstance(other, DescriptorArray):
            original_other = other.full_value.values

        if np.any(original_other == 0):
            raise ZeroDivisionError('Cannot divide by zero')
        return self._smooth_operator(other, op.truediv, units_must_match=False)
    
    def __rtruediv__(self, other: Union[DescriptorArray, DescriptorNumber, list, numbers.Number]) -> DescriptorArray:
        """
        Handle reverse division for DescriptorArrays, DescriptorNumbers, lists, and scalars.
        Ensures unit compatibility when `other` is a DescriptorNumber.
        """
        if not isinstance(other, (DescriptorArray, DescriptorNumber, list, numbers.Number)):
            return NotImplemented

        if np.any(self.full_value.values == 0):
            raise ZeroDivisionError('Cannot divide by zero')
        
        # First use __div__ to compute `self / other`
        # but first converting to the units of other
        inverse_result = self._rsmooth_operator(other, op.truediv, units_must_match=False)
        return inverse_result
    
    def __pow__(self, other: Union[DescriptorNumber, numbers.Number]) -> DescriptorArray:
        """
        Perform element-wise exponentiation with another DescriptorNumber or number.

        :param other: The object to use as a denominator. Must be a number or DescriptorNumber with
                    no unit or variance.
        :return: A new DescriptorArray representing the result of the addition.
        """
        if not isinstance(other, (numbers.Number, DescriptorNumber)):
            return NotImplemented

        if isinstance(other, numbers.Number):
            exponent = other
        elif type(other) is DescriptorNumber:
            if other.unit != 'dimensionless':
                raise UnitError('Exponents must be dimensionless')
            if other.variance is not None:
                raise ValueError('Exponents must not have variance')
            exponent = other.value
        else:
            return NotImplemented
        try:
            new_value = self.full_value**exponent
        except Exception as message:
            raise message from None
        if np.any(np.isnan(new_value.values)):
            raise ValueError('The result of the exponentiation is not a number')
        descriptor_number = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __rpow__(self, other: numbers.Number) -> numbers.Number:
        """
        Defers reverse pow with a descriptor array, `a ** array`.
        Exponentiation with regards to an array does not make sense,
        and is not implemented.
        """
        return NotImplemented

    def __neg__(self) -> DescriptorArray:
        """
        Negate all values in the DescriptorArray.
        """
        new_value = -self.full_value
        descriptor_array = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
        descriptor_array.name = descriptor_array.unique_name
        return descriptor_array

    def __abs__(self) -> DescriptorArray:
        """
        Replace all elements in the DescriptorArray with their
        absolute values. Note that this is different from the
        norm of the DescriptorArray.
        """
        new_value = abs(self.full_value)
        descriptor_array = DescriptorArray.from_scipp(name=self.name, full_value=new_value)
        descriptor_array.name = descriptor_array.unique_name
        return descriptor_array

    def __matmul__(self, other: [DescriptorArray, list]) -> DescriptorArray:
        """
        Perform matrix multiplication with with another DesciptorArray or list.

        :param other: The object to use as a denominator. Must be a DescriptorArray
                    or a list, of compatible shape.
        :return: A new DescriptorArray representing the result of the addition.
        """
        if not isinstance(other, (DescriptorArray, list)):
            return NotImplemented

        if isinstance(other, DescriptorArray):
            shape = other.full_value.shape
        elif isinstance(other, list):
            shape = np.shape(other)

        # Dimensions must match for matrix multiplication
        if shape[0] != self._array.values.shape[-1]:
            raise ValueError(f"Last dimension of {other=} must match the first dimension of DescriptorArray values")
        
        other = sc.array(dims=self._array.dims, values=other)
        new_full_value = operator(self._array, other)  # Let scipp handle operation for uncertainty propagation


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
