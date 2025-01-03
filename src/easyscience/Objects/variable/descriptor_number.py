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

from easyscience.global_object.undo_redo import property_stack_deco

from .descriptor_base import DescriptorBase


class DescriptorNumber(DescriptorBase):
    """
    A `Descriptor` for Number values with units.  The internal representation is a scipp scalar.
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
        """Constructor for the DescriptorNumber class

        param name: Name of the descriptor
        param value: Value of the descriptor
        param unit: Unit of the descriptor
        param variance: Variance of the descriptor
        param description: Description of the descriptor
        param url: URL of the descriptor
        param display_name: Display name of the descriptor
        param parent: Parent of the descriptor

        .. note:: Undo/Redo functionality is implemented for the attributes `variance` and `value`.
        """
        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise TypeError(f'{value=} must be a number')
        if variance is not None:
            if not isinstance(variance, numbers.Number) or isinstance(variance, bool):
                raise TypeError(f'{variance=} must be a number or None')
            if variance < 0:
                raise ValueError(f'{variance=} must be positive')
            variance = float(variance)
        if not isinstance(unit, sc.Unit) and not isinstance(unit, str):
            raise TypeError(f'{unit=} must be a scipp unit or a string representing a valid scipp unit')
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
            self.convert_unit(self._base_unit())

    @classmethod
    def from_scipp(cls, name: str, full_value: Variable, **kwargs) -> DescriptorNumber:
        """
        Create a DescriptorNumber from a scipp constant.

        :param name: Name of the descriptor
        :param value: Value of the descriptor as a scipp scalar
        :param kwargs: Additional parameters for the descriptor
        :return: DescriptorNumber
        """
        if not isinstance(full_value, Variable):
            raise TypeError(f'{full_value=} must be a scipp scalar')
        if len(full_value.dims) != 0:
            raise TypeError(f'{full_value=} must be a scipp scalar')
        return cls(name=name, value=full_value.value, unit=full_value.unit, variance=full_value.variance, **kwargs)

    @property
    def full_value(self) -> Variable:
        """
        Get the value of self as a scipp scalar. This is should be usable for most cases.

        :return: Value of self with unit.
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
        Get the value. This should be usable for most cases. The full value can be obtained from `obj.full_value`.

        :return: Value of self with unit.
        """
        return self._scalar.value

    @value.setter
    @property_stack_deco
    def value(self, value: numbers.Number) -> None:
        """
        Set the value of self. This should be usable for most cases. The full value can be obtained from `obj.full_value`.

        :param value: New value of self
        """
        if not isinstance(value, numbers.Number) or isinstance(value, bool):
            raise TypeError(f'{value=} must be a number')
        self._scalar.value = float(value)

    @property
    def unit(self) -> str:
        """
        Get the unit.

        :return: Unit as a string.
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

        :return: variance.
        """
        return self._scalar.variance

    @variance.setter
    @property_stack_deco
    def variance(self, variance_float: float) -> None:
        """
        Set the variance.

        :param variance_float: Variance as a float
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

        :return: Error associated with parameter
        """
        return float(np.sqrt(self._scalar.variance))

    @error.setter
    @property_stack_deco
    def error(self, value: float) -> None:
        """
        Set the standard deviation for the parameter.

        :param value: New error value
        """
        if value is not None:
            if not isinstance(value, numbers.Number):
                raise TypeError(f'{value=} must be a number or None')
            if value < 0:
                raise ValueError(f'{value=} must be positive')
            value = float(value)
        self._scalar.variance = value**2

    def convert_unit(self, unit_str: str):
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
        self._scalar = self._scalar.to(unit=new_unit)

    # Just to get return type right
    def __copy__(self) -> DescriptorNumber:
        return super().__copy__()

    def __repr__(self) -> str:
        """Return printable representation."""
        string = '<'
        string += self.__class__.__name__ + ' '
        string += f"'{self._name}': "
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
        return raw_dict

    def __add__(self, other: Union[DescriptorNumber, numbers.Number]) -> DescriptorNumber:
        if isinstance(other, numbers.Number):
            if self.unit != 'dimensionless':
                raise UnitError('Numbers can only be added to dimensionless values')
            new_value = self.full_value + other
        elif type(other) is DescriptorNumber:
            original_unit = other.unit
            try:
                other.convert_unit(self.unit)
            except UnitError:
                raise UnitError(f'Values with units {self.unit} and {other.unit} cannot be added') from None
            new_value = self.full_value + other.full_value
            other.convert_unit(original_unit)
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
                other.convert_unit(self.unit)
            except UnitError:
                raise UnitError(f'Values with units {self.unit} and {other.unit} cannot be subtracted') from None
            new_value = self.full_value - other.full_value
            other.convert_unit(original_unit)
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
        descriptor_number.convert_unit(descriptor_number._base_unit())
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
            original_other = other
            if other == 0:
                raise ZeroDivisionError('Cannot divide by zero')
            new_value = self.full_value / other
        elif type(other) is DescriptorNumber:
            original_other = other.value
            if original_other == 0:
                raise ZeroDivisionError('Cannot divide by zero')
            new_value = self.full_value / other.full_value
            other.value = original_other
        else:
            return NotImplemented
        descriptor_number = DescriptorNumber.from_scipp(name=self.name, full_value=new_value)
        descriptor_number.convert_unit(descriptor_number._base_unit())
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
        string = str(self._scalar.unit)
        for i, letter in enumerate(string):
            if letter == 'e':
                if string[i : i + 2] not in ['e+', 'e-']:
                    return string[i:]
            elif letter not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '+', '-']:
                return string[i:]
        return ''
