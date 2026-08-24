# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers
import uuid
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import scipp as sc
from scipp import UnitError
from scipp import Variable

from easyscience.global_object.undo_redo import PropertyStack
from easyscience.global_object.undo_redo import property_stack

from .descriptor_base import DescriptorBase


# Why is this a decorator? Because otherwise we would need a flag on the convert_unit method to avoid
# infinite recursion. This is a bit cleaner as it avoids the need for a internal only flag on a user method.
def notify_observers(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to notify observers of a change in the descriptor.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to be decorated.

    Returns
    -------
    Callable[..., Any]
        Decorated function.
    """

    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._notify_observers()
        return result

    return wrapper


class DescriptorNumber(DescriptorBase):
    """
    A ``Descriptor`` for Number values with units.

    The internal representation is a scipp scalar.
    """

    # Tell ``SerializerBase._recursive_encoder`` to serialize nested instances
    # through ``as_dict`` so the serializer id (and, for ``Parameter``, the
    # dependency expression) is not lost when this object is a child of
    # another object.
    _encode_with_as_dict = True

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
        **kwargs: Any,  # Additional keyword arguments (used for (de)serialization)
    ):
        """
        Constructor for the DescriptorNumber class.

        param name: Name of the descriptor param value: Value of the
        descriptor param unit: Unit of the descriptor param variance:
        Variance of the descriptor param description: Description of the
        descriptor param url: URL of the descriptor param display_name:
        Display name of the descriptor param parent: Parent of the
        descriptor .. note:: Undo/Redo functionality is implemented for
        the attributes ``variance``, ``error``, ``unit`` and ``value``.
        """
        self._observers: List[DescriptorNumber] = []

        # Extract serializer_id if provided during deserialization
        if '__serializer_id' in kwargs:
            self.__serializer_id = kwargs.pop('__serializer_id')

        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise TypeError(f'{value=} must be a number')
        if variance is not None:
            if not isinstance(variance, numbers.Number) or isinstance(variance, bool):
                raise TypeError(f'{variance=} must be a number or None')
            if variance < 0:
                raise ValueError(f'{variance=} must be positive')
            variance = float(variance)
        if not isinstance(unit, sc.Unit) and not isinstance(unit, str):
            raise TypeError(
                f'{unit=} must be a scipp unit or a string representing a valid scipp unit'
            )
        try:
            self._scalar = sc.scalar(float(value), unit=unit, variance=variance)
        except Exception as message:
            raise UnitError(message)
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
            self._convert_unit(self._base_unit())

    @classmethod
    def from_scipp(cls, name: str, full_value: Variable, **kwargs: Any) -> DescriptorNumber:
        """
        Create a DescriptorNumber from a scipp constant.

        Parameters
        ----------
        name : str
            Name of the descriptor.
        full_value : Variable
            Value of the descriptor as a scipp scalar.
        **kwargs : Any
            Additional parameters for the descriptor.

        Returns
        -------
        DescriptorNumber
            DescriptorNumber.

        Raises
        ------
        TypeError
            If ``full_value`` is not a scalar scipp ``Variable``.
        """
        if not isinstance(full_value, Variable):
            raise TypeError(f'{full_value=} must be a scipp scalar')
        if len(full_value.dims) != 0:
            raise TypeError(f'{full_value=} must be a scipp scalar')
        return cls(
            name=name,
            value=full_value.value,
            unit=full_value.unit,
            variance=full_value.variance,
            **kwargs,
        )

    def _attach_observer(self, observer: DescriptorNumber) -> None:
        """Attach an observer to the descriptor."""
        self._observers.append(observer)
        if not hasattr(self, '_DescriptorNumber__serializer_id'):
            self.__serializer_id = str(uuid.uuid4())

    def _detach_observer(self, observer: DescriptorNumber) -> None:
        """Detach an observer from the descriptor."""
        self._observers.remove(observer)
        if not self._observers:
            del self.__serializer_id

    def _notify_observers(self) -> None:
        """Notify all observers of a change."""
        for observer in self._observers:
            observer._update()

    def _validate_dependencies(self, origin: Optional[str] = None) -> None:
        """
        Ping all observers to check if any cyclic dependencies have been
        introduced.

        Parameters
        ----------
        origin : Optional[str], default=None
            Unique_name of the origin of this validation check. Used to
            avoid cyclic depenencies. By default, None.

        Raises
        ------
        RuntimeError
            If a cyclic dependency is detected.
        """
        if origin == self.unique_name:
            raise RuntimeError(
                '\n Cyclic dependency detected!\n'
                + f'An update of {self.unique_name} leads to it updating itself.\n'
                + 'Please check your dependencies.'
            )
        if origin is None:
            origin = self.unique_name
        for observer in self._observers:
            observer._validate_dependencies(origin=origin)

    @property
    def full_value(self) -> Variable:
        """
        Get the value of self as a scipp scalar.

        This is should be usable for most cases.

        Returns
        -------
        Variable
            Value of self with unit.
        """
        return self._scalar

    @full_value.setter
    def full_value(self, full_value: Variable) -> None:
        raise AttributeError(
            f'Full_value is read-only. Change the value and variance seperately. Or create a new {self.__class__.__name__}.'
        )

    @property
    def value(self) -> numbers.Number:
        """
        Get the value.

        This should be usable for most cases. The full value can be
        obtained from ``obj.full_value``.

        Returns
        -------
        numbers.Number
            Value of self with unit.
        """
        return self._scalar.value

    @value.setter
    @notify_observers
    @property_stack
    def value(self, value: numbers.Number) -> None:
        """
        Set the value of self.

        This should be usable for most cases. The full value can be
        obtained from ``obj.full_value``.

        Parameters
        ----------
        value : numbers.Number
            New value of self.

        Raises
        ------
        TypeError
            If ``value`` is not a number.
        """
        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise TypeError(f'{value=} must be a number')
        self._scalar.value = float(value)

    @property
    def unit(self) -> str:
        """
        Get the unit.

        Returns
        -------
        str
            Unit as a string.
        """
        return str(self._scalar.unit)

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

        Returns
        -------
        float
            Variance.
        """
        return self._scalar.variance

    @variance.setter
    @notify_observers
    @property_stack
    def variance(self, variance_float: float) -> None:
        """
        Set the variance.

        Parameters
        ----------
        variance_float : float
            Variance as a float.

        Raises
        ------
        TypeError
            If ``variance_float`` is not numeric.
        ValueError
            If ``variance_float`` is negative.
        """
        if variance_float is not None:
            if not isinstance(variance_float, numbers.Number):
                raise TypeError(f'{variance_float=} must be a number or None')
            if variance_float < 0:
                raise ValueError(f'{variance_float=} must be positive')
            variance_float = float(variance_float)
        self._scalar.variance = variance_float

    @property
    def error(self) -> float:
        """
        The standard deviation for the parameter.

        Returns
        -------
        float
            Error associated with parameter.
        """
        if self._scalar.variance is None:
            return None
        return float(np.sqrt(self._scalar.variance))

    @error.setter
    @notify_observers
    @property_stack
    def error(self, value: float) -> None:
        """
        Set the standard deviation for the parameter.

        Parameters
        ----------
        value : float
            New error value.

        Raises
        ------
        TypeError
            If ``value`` is not numeric.
        ValueError
            If ``value`` is negative.
        """
        if value is not None:
            if not isinstance(value, numbers.Number):
                raise TypeError(f'{value=} must be a number or None')
            if value < 0:
                raise ValueError(f'{value=} must be positive')
            value = float(value)
            self._scalar.variance = value**2
        else:
            self._scalar.variance = None

    # When we convert units internally, we dont want to notify observers as this can cause infinite recursion.
    # Therefore the convert_unit method is split into two methods, a private internal method and a public method.
    def _convert_unit(self, unit_str: str) -> None:
        """
        Convert the value from one unit system to another.

        Parameters
        ----------
        unit_str : str
            New unit in string form.

        Raises
        ------
        TypeError
            If ``unit_str`` is not a string.
        UnitError
            If the unit conversion fails.
        """
        if not isinstance(unit_str, str):
            raise TypeError(f'{unit_str=} must be a string representing a valid scipp unit')
        new_unit = sc.Unit(unit_str)

        # Save the current state for undo/redo
        old_scalar = self._scalar

        # Perform the unit conversion
        try:
            new_scalar = self._scalar.to(unit=new_unit)
        except Exception as e:
            raise UnitError(f'Failed to convert unit: {e}') from e

        # Define the setter function for the undo stack
        def set_scalar(obj, scalar):
            obj._scalar = scalar

        # Push to undo stack
        self._global_object.stack.push(
            PropertyStack(
                self, set_scalar, old_scalar, new_scalar, text=f'Convert unit to {unit_str}'
            )
        )

        # Update the scalar
        self._scalar = new_scalar

    # When the user calls convert_unit, we want to notify observers of the change to propagate the change.
    @notify_observers
    def convert_unit(self, unit_str: str) -> None:
        """
        Convert the value from one unit system to another.

        Parameters
        ----------
        unit_str : str
            New unit in string form.
        """
        self._convert_unit(unit_str)

    # Just to get return type right
    def __copy__(self) -> DescriptorNumber:
        return super().__copy__()

    def __repr__(self) -> str:
        """Return printable representation."""
        string = '<'
        string += self.__class__.__name__ + ' '
        string += f"'{self._name}': "
        if np.abs(self._scalar.value) > 1e4 or (
            np.abs(self._scalar.value) < 1e-4 and self._scalar.value != 0
        ):
            # Use scientific notation for large or small values
            string += f'{self._scalar.value:.3e}'
            if self.variance:
                string += f' \u00b1 {self.error:.3e}'
        else:
            string += f'{self._scalar.value:.4f}'
            if self.variance:
                string += f' \u00b1 {self.error:.4f}'
        obj_unit = self._scalar.unit
        if obj_unit == 'dimensionless':
            obj_unit = ''
        else:
            obj_unit = f' {obj_unit}'
        string += obj_unit
        string += '>'
        return string
        # return f"<{class_name} '{obj_name}': {obj_value:0.04f}{obj_unit}>"

    def as_dict(self, skip: Optional[List[str]] = None) -> Dict[str, Any]:
        raw_dict = super().as_dict(skip=skip)
        raw_dict['value'] = self._scalar.value
        raw_dict['unit'] = str(self._scalar.unit)
        raw_dict['variance'] = self._scalar.variance
        if hasattr(self, '_DescriptorNumber__serializer_id'):
            raw_dict['__serializer_id'] = self.__serializer_id
        return raw_dict

    def __add__(self, other: Union[DescriptorNumber, numbers.Number]) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError('Numbers can only be added to dimensionless values')
            new_value = self.full_value + other
        elif type(other) is DescriptorNumber:
            original_unit = other.unit
            try:
                other._convert_unit(self.unit)
            except UnitError:
                raise UnitError(
                    f'Values with units {self.unit} and {other.unit} cannot be added'
                ) from None
            new_value = self.full_value + other.full_value
            other._convert_unit(original_unit)
        else:
            return NotImplemented
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __radd__(self, other: numbers.Number) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError('Numbers can only be added to dimensionless values')
            new_value = other + self.full_value
        else:
            return NotImplemented
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __sub__(self, other: Union[DescriptorNumber, numbers.Number]) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError('Numbers can only be subtracted from dimensionless values')
            new_value = self.full_value - other
        elif type(other) is DescriptorNumber:
            original_unit = other.unit
            try:
                other._convert_unit(self.unit)
            except UnitError:
                raise UnitError(
                    f'Values with units {self.unit} and {other.unit} cannot be subtracted'
                ) from None
            new_value = self.full_value - other.full_value
            other._convert_unit(original_unit)
        else:
            return NotImplemented
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __rsub__(self, other: numbers.Number) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError('Numbers can only be subtracted from dimensionless values')
            new_value = other - self.full_value
        else:
            return NotImplemented
        descriptor = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor.name = descriptor.unique_name
        return descriptor

    def __mul__(self, other: Union[DescriptorNumber, numbers.Number]) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            new_value = self.full_value * other
        elif type(other) is DescriptorNumber:
            new_value = self.full_value * other.full_value
        else:
            return NotImplemented
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number._convert_unit(descriptor_number._base_unit())
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __rmul__(self, other: numbers.Number) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            new_value = other * self.full_value
        else:
            return NotImplemented
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __truediv__(self, other: Union[DescriptorNumber, numbers.Number]) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            if other == 0:
                raise ZeroDivisionError('Cannot divide by zero')
            new_value = self.full_value / other
        elif type(other) is DescriptorNumber:
            if other.value == 0:
                raise ZeroDivisionError('Cannot divide by zero')
            new_value = self.full_value / other.full_value
        else:
            return NotImplemented
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number._convert_unit(descriptor_number._base_unit())
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __rtruediv__(self, other: numbers.Number) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            if self.value == 0:
                raise ZeroDivisionError('Cannot divide by zero')
            new_value = other / self.full_value
        else:
            return NotImplemented
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __pow__(self, other: Union[DescriptorNumber, numbers.Number]) -> DescriptorNumber:
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
        if np.isnan(new_value.value):
            raise ValueError('The result of the exponentiation is not a number')
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __rpow__(self, other: numbers.Number) -> numbers.Number:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError('Exponents must be dimensionless')
            if self.variance is not None:
                raise ValueError('Exponents must not have variance')
            new_value = other**self.value
        else:
            return NotImplemented
        return new_value

    def __neg__(self) -> DescriptorNumber:
        new_value = -self.full_value
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def __abs__(self) -> DescriptorNumber:
        new_value = abs(self.full_value)
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.name = descriptor_number.unique_name
        return descriptor_number

    def _base_unit(self) -> str:
        """
        Extract the base unit from the unit string by removing numeric
        components and scientific notation.
        """
        string = str(self._scalar.unit)
        for i, letter in enumerate(string):
            if letter == 'e':
                if string[i : i + 2] not in ['e+', 'e-']:
                    return string[i:]
            elif letter not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '+', '-']:
                return string[i:]
        return ''
