#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import numbers
import warnings
import weakref
from copy import deepcopy
from inspect import getfullargspec
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np

from easyscience import global_object
from easyscience import pint
from easyscience import ureg
from easyscience.Constraints import SelfConstraint
from easyscience.global_object.undo_redo import property_stack_deco
from easyscience.Objects.core import ComponentSerializer
from easyscience.Utils.classTools import addProp
from easyscience.Utils.Exceptions import CoreSetException

if TYPE_CHECKING:
    from easyscience.Constraints import C

Q_ = ureg.Quantity
M_ = ureg.Measurement


class Descriptor(ComponentSerializer):
    """
    This is the base of all variable descriptions for models. It contains all information to describe a single
    unique property of an object. This description includes a name and value as well as optionally a unit, description
    and url (for reference material). Also implemented is a callback so that the value can be read/set from a linked
    library object.

    A `Descriptor` is typically something which describes part of a model and is non-fittable and generally changes the
    state of an object.
    """

    _constructor = Q_
    _global_object = global_object
    _REDIRECT = {
        'value': lambda obj: obj.raw_value,
        'units': lambda obj: obj._args['units'],
        'parent': None,
        'callback': None,
        '_finalizer': None,
    }

    def __init__(
        self,
        name: str,
        value: Any,
        units: Optional[Union[str, ureg.Unit]] = None,
        unique_name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        display_name: Optional[str] = None,
        callback: Optional[property] = property(),
        enabled: Optional[bool] = True,
        parent: Optional[Union[Any, None]] = None,
    ):  # noqa: S107
        """
        This is the base of all variable descriptions for models. It contains all information to describe a single
        unique property of an object. This description includes a name and value as well as optionally a unit,
        description and url (for reference material). Also implemented is a callback so that the value can be read/set
        from a linked library object.

        A `Descriptor` is typically something which describes part of a model and is non-fittable and generally changes
        the state of an object.

        Units are provided by pint: https://github.com/hgrecco/pint

        :param name: Name of this object
        :param value: Value of this object
        :param units: This object can have a physical unit associated with it
        :param description: A brief summary of what this object is
        :param url: Lookup url for documentation/information
        :param callback: The property which says how the object is linked to another one
        :param parent: The object which is the parent to this one

        .. code-block:: python

             from easyscience.Objects.Base import Descriptor
             # Describe a color by text
             color_text = Descriptor('fav_colour', 'red')
             # Describe a color by RGB
             color_num = Descriptor('fav_colour', [1, 0, 0])

        .. note:: Undo/Redo functionality is implemented for the attributes `value`, `unit` and `display name`.
        """
        if not hasattr(self, '_args'):
            self._args = {'value': None, 'units': ''}
        if unique_name is None:
            unique_name = self._global_object.generate_unique_name(self.__class__.__name__)
        self._unique_name = unique_name
        self.name = name
        # Let the collective know we've been assimilated
        self._global_object.map.add_vertex(self, obj_type='created')
        # Make the connection between self and parent
        if parent is not None:
            self._global_object.map.add_edge(parent, self)

        # Attach units if necessary
        if isinstance(units, ureg.Unit):
            self._units = ureg.Quantity(1, units=deepcopy(units))
        elif isinstance(units, (str, type(None))):
            self._units = ureg.parse_expression(units)
        else:
            raise AttributeError('Units must be a string or a pint unit object')
        # Clunky method of keeping self.value up to date
        self._type = type(value)
        self.__isBooleanValue = isinstance(value, bool)
        if self.__isBooleanValue:
            value = int(value)
        self._args['value'] = value
        self._args['units'] = str(self.unit)
        self._value = self.__class__._constructor(**self._args)

        self._enabled = enabled

        if description is None:
            description = ''
        self.description: str = description

        self._display_name: str = display_name

        if url is None:
            url = ''

        self.url: str = url
        if callback is None:
            callback = property()
        self._callback: property = callback
        self.user_data: dict = {}

        finalizer = None
        if self._callback.fdel is not None:
            weakref.finalize(self, self._callback.fdel)
        self._finalizer = finalizer

    @property
    def _arg_spec(self) -> Set[str]:
        base_cls = getattr(self, '__old_class__', self.__class__)
        mro = base_cls.__mro__
        idx = mro.index(ComponentSerializer)
        names = set()
        for i in range(idx):
            cls = mro[i]
            if hasattr(cls, '_CORE'):
                spec = getfullargspec(cls.__init__)
                names = names.union(set(spec.args[1:]))
        return names

    def __reduce__(self):
        """
        Make the class picklable. Due to the nature of the dynamic class definitions special measures need to be taken.

        :return: Tuple consisting of how to make the object
        :rtype: tuple
        """
        state = self.encode()
        cls = self.__class__
        if hasattr(self, '__old_class__'):
            cls = self.__old_class__
        return cls.from_dict, (state,)

    @property
    def unique_name(self) -> str:
        """
        Get the unique name of this object.

        :return: Unique name of this object
        """
        return self._unique_name

    @unique_name.setter
    def unique_name(self, new_unique_name: str):
        """Set a new unique name for the object. The old name is still kept in the map.

        :param new_unique_name: New unique name for the object"""
        if not isinstance(new_unique_name, str):
            raise TypeError('Unique name has to be a string.')
        self._unique_name = new_unique_name
        self._global_object.map.add_vertex(self)

    @property
    def display_name(self) -> str:
        """
        Get a pretty display name.

        :return: The pretty display name.
        """
        # TODO This might be better implementing fancy f-strings where we can return html,latex, markdown etc
        display_name = self._display_name
        if display_name is None:
            display_name = self.name
        return display_name

    @display_name.setter
    @property_stack_deco
    def display_name(self, name_str: str):
        """
        Set the pretty display name.

        :param name_str: Pretty display name of the object.
        :return: None
        """
        self._display_name = name_str

    @property
    def unit(self) -> pint.UnitRegistry:
        """
        Get the unit associated with the object.

        :return: Unit associated with self in `pint` form.
        """
        return self._units.units

    @unit.setter
    @property_stack_deco
    def unit(self, unit_str: str):
        """
        Set the unit to a new one.

        :param unit_str: String representation of the unit required. i.e `m/s`
        :return: None
        """
        if not isinstance(unit_str, str):
            unit_str = str(unit_str)
        new_unit = ureg.parse_expression(unit_str)
        self._units = new_unit
        self._args['units'] = str(new_unit)
        self._value = self.__class__._constructor(**self._args)

    @property
    def value(self) -> Any:
        """
        Get the value of self as a pint. This is should be usable for most cases. If a pint
        is not acceptable then the raw value can be obtained through `obj.raw_value`.

        :return: Value of self with unit.
        """
        # Cached property? Should reference callback.
        # Also should reference for undo/redo
        if self._callback.fget is not None:
            try:
                value = self._callback.fget()
                if hasattr(self._value, 'magnitude'):
                    if value != self._value.magnitude:
                        self.__deepValueSetter(value)
                elif value != self._value:
                    self.__deepValueSetter(value)

            except Exception as e:
                raise ValueError(f'Unable to return value:\n{e}')
        r_value = self._value
        if self.__isBooleanValue:
            r_value = bool(r_value)
        return r_value

    def __deepValueSetter(self, value: Any):
        """
        Set the value of self. This creates a pint with a unit.

        :param value: New value of self
        :return: None
        """
        # TODO there should be a callback to the collective, logging this as a return(if from a non `EasyScience` class)
        if hasattr(value, 'magnitude'):
            value = value.magnitude
            if hasattr(value, 'nominal_value'):
                value = value.nominal_value
        self._type = type(value)
        self.__isBooleanValue = isinstance(value, bool)
        if self.__isBooleanValue:
            value = int(value)
        self._args['value'] = value
        self._value = self.__class__._constructor(**self._args)

    @value.setter
    @property_stack_deco
    def value(self, value: Any):
        """
        Set the value of self. This creates a pint with a unit.

        :param value: New value of self
        :return: None
        """
        self.__deepValueSetter(value)
        if self._callback.fset is not None:
            try:
                self._callback.fset(value)
            except Exception as e:
                raise CoreSetException(e)

    @property
    def raw_value(self) -> Any:
        """
        Return the raw value of self without a unit.

        :return: The raw value of self
        """
        value = self._value
        if hasattr(value, 'magnitude'):
            value = value.magnitude
            if hasattr(value, 'nominal_value'):
                value = value.nominal_value
        if self.__isBooleanValue:
            value = bool(value)
        return value

    @property
    def enabled(self) -> bool:
        """
        Logical property to see if the objects value can be directly set.

        :return: Can the objects value be set
        """
        return self._enabled

    @enabled.setter
    @property_stack_deco
    def enabled(self, value: bool):
        """
        Enable and disable the direct setting of an objects value field.

        :param value: True - objects value can be set, False - the opposite
        """
        self._enabled = value

    def convert_unit(self, unit_str: str):
        """
        Convert the value from one unit system to another. You will should use
        `compatible_units` to see if your new unit is compatible.

        :param unit_str: New unit in string form
        """
        new_unit = ureg.parse_expression(unit_str)
        self._value = self._value.to(new_unit)
        self._units = new_unit
        self._args['value'] = self.raw_value
        self._args['units'] = str(self.unit)

    # @cached_property
    @property
    def compatible_units(self) -> List[str]:
        """
        Returns all possible units for which the current unit can be converted.

        :return: Possible conversion units
        """
        return [str(u) for u in self.unit.compatible_units()]

    def __repr__(self):
        """Return printable representation of a Descriptor/Parameter object."""
        class_name = self.__class__.__name__
        obj_name = self.name
        if self.__isBooleanValue:
            obj_value = self.raw_value
        else:
            obj_value = self._value.magnitude
        if isinstance(obj_value, float):
            obj_value = '{:0.04f}'.format(obj_value)
        obj_units = ''
        if not self.unit.dimensionless:
            obj_units = ' {:~P}'.format(self.unit)
        out_str = f"<{class_name} '{obj_name}': {obj_value}{obj_units}>"
        return out_str

    def to_obj_type(self, data_type: Type[Parameter], *kwargs):
        """
        Convert between a `Parameter` and a `Descriptor`.

        :param data_type: class constructor of what we want to be
        :param kwargs: Additional keyword/value pairs for conversion
        :return: self as a new type
        """
        pickled_obj = self.encode()
        pickled_obj.update(kwargs)
        if '@class' in pickled_obj.keys():
            pickled_obj['@class'] = data_type.__name__
        return data_type.from_dict(pickled_obj)

    def __copy__(self):
        return self.__class__.from_dict(self.as_dict())


V = TypeVar('V', bound=Descriptor)


class ComboDescriptor(Descriptor):
    """
    This class is an extension of a ``EasyScience.Object.Base.Descriptor``. This class has a selection of values which can
    be checked against. For example, combo box styling.
    """

    def __init__(self, *args, available_options: list = None, **kwargs):
        super(ComboDescriptor, self).__init__(*args, **kwargs)
        if available_options is None:
            available_options = []
        self._available_options = available_options

        # We have initialized from the Descriptor class where value has it's own undo/redo decorator
        # This needs to be bypassed to use the Parameter undo/redo stack
        fun = self.__class__.value.fset
        if hasattr(fun, 'func'):
            fun = getattr(fun, 'func')
        self.__previous_set: Callable[[V, Union[numbers.Number, np.ndarray]], Union[numbers.Number, np.ndarray]] = fun

        # Monkey patch the unit and the value to take into account the new max/min situation
        addProp(
            self,
            'value',
            fget=self.__class__.value.fget,
            fset=self.__class__._property_value.fset,
            fdel=self.__class__.value.fdel,
        )

    @property
    def _property_value(self) -> Union[numbers.Number, np.ndarray]:
        return self.value

    @_property_value.setter
    @property_stack_deco
    def _property_value(self, set_value: Union[numbers.Number, np.ndarray, Q_]):
        """
        Verify value against constraints. This hasn't really been implemented as fitting is tricky.

        :param set_value: value to be verified
        :return: new value from constraint
        """
        if isinstance(set_value, Q_):
            set_value = set_value.magnitude
        # Save the old state and create the new state
        old_value = self._value
        state = self._global_object.stack.enabled
        if state:
            self._global_object.stack.force_state(False)
        try:
            new_value = old_value
            if set_value in self.available_options:
                new_value = set_value
        finally:
            self._global_object.stack.force_state(state)

        # Restore to the old state
        self.__previous_set(self, new_value)

    @property
    def available_options(self) -> List[Union[numbers.Number, np.ndarray, Q_]]:
        return self._available_options

    @available_options.setter
    @property_stack_deco
    def available_options(self, available_options: List[Union[numbers.Number, np.ndarray, Q_]]) -> None:
        self._available_options = available_options

    def as_dict(self, **kwargs) -> Dict[str, Any]:
        import json

        d = super().as_dict(**kwargs)
        d['name'] = self.name
        d['available_options'] = json.dumps(self.available_options)
        return d


class Parameter(Descriptor):
    """
    This class is an extension of a ``EasyScience.Object.Base.Descriptor``. Where the descriptor was for static objects,
    a `Parameter` is for dynamic objects. A parameter has the ability to be used in fitting and
    has additional fields to facilitate this.
    """

    _constructor = M_

    def __init__(
        self,
        name: str,
        value: Union[numbers.Number, np.ndarray],
        error: Optional[Union[numbers.Number, np.ndarray]] = 0.0,
        min: Optional[numbers.Number] = -np.inf,
        max: Optional[numbers.Number] = np.inf,
        fixed: Optional[bool] = False,
        **kwargs,
    ):
        """
        This class is an extension of a ``EasyScience.Object.Base.Descriptor``. Where the descriptor was for static
        objects, a `Parameter` is for dynamic objects. A parameter has the ability to be used in fitting and has
        additional fields to facilitate this.

        :param name: Name of this obj
        :param value: Value of this object
        :param error: Error associated as sigma for this parameter
        :param min: Minimum value for fitting
        :param max: Maximum value for fitting
        :param fixed: Should this parameter vary when fitting?
        :param kwargs: Key word arguments for the `Descriptor` class.

        .. code-block:: python

             from easyscience.Objects.Base import Parameter
             # Describe a phase
             phase_basic = Parameter('phase', 3)
             # Describe a phase with a unit
             phase_unit = Parameter('phase', 3, units,='rad/s')

        .. note::
            Undo/Redo functionality is implemented for the attributes `value`, `error`, `min`, `max`, `fixed`
        """
        # Set the error
        self._args = {'value': value, 'units': '', 'error': error}

        if not isinstance(value, numbers.Number) or isinstance(value, np.ndarray):
            raise ValueError('In a parameter the `value` must be numeric')
        if value < min:
            raise ValueError('`value` can not be less than `min`')
        if value > max:
            raise ValueError('`value` can not be greater than `max`')
        if error < 0:
            raise ValueError('Standard deviation `error` must be positive')

        super().__init__(name=name, value=value, **kwargs)
        self._args['units'] = str(self.unit)

        # Warnings if we are given a boolean
        if isinstance(self._type, bool):
            warnings.warn(
                'Boolean values are not officially supported in Parameter. Use a Descriptor instead',
                UserWarning,
            )

        # Create additional fitting elements
        self._min: numbers.Number = min
        self._max: numbers.Number = max
        self._fixed: bool = fixed
        self.initial_value = self.value
        self._constraints: dict = {
            'user': {},
            'builtin': {
                'min': SelfConstraint(self, '>=', '_min'),
                'max': SelfConstraint(self, '<=', '_max'),
            },
            'virtual': {},
        }
        # This is for the serialization. Otherwise we wouldn't catch the values given to `super()`
        self._kwargs = kwargs

        # We have initialized from the Descriptor class where value has it's own undo/redo decorator
        # This needs to be bypassed to use the Parameter undo/redo stack
        fun = self.__class__.value.fset
        if hasattr(fun, 'func'):
            fun = getattr(fun, 'func')
        self.__previous_set: Callable[
            [V, Union[numbers.Number, np.ndarray]],
            Union[numbers.Number, np.ndarray],
        ] = fun

        # Monkey patch the unit and the value to take into account the new max/min situation
        addProp(
            self,
            'value',
            fget=self.__class__.value.fget,
            fset=self.__class__._property_value.fset,
            fdel=self.__class__.value.fdel,
        )

    @property
    def _property_value(self) -> Union[numbers.Number, np.ndarray, M_]:
        return self.value

    @_property_value.setter
    @property_stack_deco
    def _property_value(self, set_value: Union[numbers.Number, np.ndarray, M_]) -> None:
        """
        Verify value against constraints. This hasn't really been implemented as fitting is tricky.

        :param set_value: value to be verified
        :return: new value from constraint
        """
        if isinstance(set_value, M_):
            set_value = set_value.magnitude.nominal_value
        # Save the old state and create the new state
        old_value = self._value
        self._value = self.__class__._constructor(value=set_value, units=self._args['units'], error=self._args['error'])

        # First run the built in constraints. i.e. min/max
        constraint_type: MappingProxyType[str, C] = self.builtin_constraints
        new_value = self.__constraint_runner(constraint_type, set_value)
        # Then run any user constraints.
        constraint_type: dict = self.user_constraints
        state = self._global_object.stack.enabled
        if state:
            self._global_object.stack.force_state(False)
        try:
            new_value = self.__constraint_runner(constraint_type, new_value)
        finally:
            self._global_object.stack.force_state(state)

        # And finally update any virtual constraints
        constraint_type: dict = self._constraints['virtual']
        _ = self.__constraint_runner(constraint_type, new_value)

        # Restore to the old state
        self._value = old_value
        self.__previous_set(self, new_value)

    def convert_unit(self, new_unit: str):  # noqa: S1144
        """
        Perform unit conversion. The value, max and min can change on unit change.

        :param new_unit: new unit
        :return: None
        """
        old_unit = str(self._args['units'])
        super().convert_unit(new_unit)
        # Deal with min/max. Error is auto corrected
        if not self.value.unitless and old_unit != 'dimensionless':
            self._min = Q_(self.min, old_unit).to(self._units).magnitude
            self._max = Q_(self.max, old_unit).to(self._units).magnitude
        # Log the new converted error
        self._args['error'] = self.value.error.magnitude

    @property
    def min(self) -> numbers.Number:
        """
        Get the minimum value for fitting.

        :return: minimum value
        """
        return self._min

    @min.setter
    @property_stack_deco
    def min(self, value: numbers.Number):
        """
        Set the minimum value for fitting.
        - implements undo/redo functionality.

        :param value: new minimum value
        :return: None
        """
        if value <= self.raw_value:
            self._min = value
        else:
            raise ValueError(f'The current set value ({self.raw_value}) is less than the desired min value ({value}).')

    @property
    def max(self) -> numbers.Number:
        """
        Get the maximum value for fitting.

        :return: maximum value
        """
        return self._max

    @max.setter
    @property_stack_deco
    def max(self, value: numbers.Number):
        """
        Get the maximum value for fitting.
        - implements undo/redo functionality.

        :param value: new maximum value
        :return: None
        """
        if value >= self.raw_value:
            self._max = value
        else:
            raise ValueError(f'The current set value ({self.raw_value}) is greater than the desired max value ({value}).')

    @property
    def fixed(self) -> bool:
        """
        Can the parameter vary while fitting?

        :return: True = fixed, False = can vary
        :rtype: bool
        """
        return self._fixed

    @fixed.setter
    @property_stack_deco
    def fixed(self, value: bool):
        """
        Change the parameter vary while fitting state.
        - implements undo/redo functionality.

        :param value: True = fixed, False = can vary
        :return: None
        """
        if not self.enabled:
            if self._global_object.stack.enabled:
                self._global_object.stack.pop()
            if global_object.debug:
                raise CoreSetException(f'{str(self)} is not enabled.')
            return
        # TODO Should we try and cast value to bool rather than throw ValueError?
        if not isinstance(value, bool):
            raise ValueError
        self._fixed = value

    @property
    def free(self) -> bool:
        return not self.fixed

    @free.setter
    def free(self, value: bool) -> None:
        self.fixed = not value

    @property
    def error(self) -> float:
        """
        The error associated with the parameter.

        :return: Error associated with parameter
        """
        return float(self._value.error.magnitude)

    @error.setter
    @property_stack_deco
    def error(self, value: float):
        """
        Set the error associated with the parameter.
        - implements undo/redo functionality.

        :param value: New error value
        :return: None
        """
        if value < 0:
            raise ValueError
        self._args['error'] = value
        self._value = self.__class__._constructor(**self._args)

    def __repr__(self) -> str:
        """
        Return printable representation of a Parameter object.
        """
        super_str = super().__repr__()
        super_str = super_str[:-1]
        s = []
        if self.fixed:
            super_str += ' (fixed)'
        s.append(super_str)
        s.append('bounds=[%s:%s]' % (repr(self.min), repr(self.max)))
        return '%s>' % ', '.join(s)

    def __float__(self) -> float:
        return float(self.raw_value)

    @property
    def builtin_constraints(self) -> MappingProxyType[str, C]:
        """
        Get the built in constrains of the object. Typically these are the min/max

        :return: Dictionary of constraints which are built into the system
        """
        return MappingProxyType(self._constraints['builtin'])

    @property
    def user_constraints(self) -> Dict[str, C]:
        """
        Get the user specified constrains of the object.

        :return: Dictionary of constraints which are user supplied
        """
        return self._constraints['user']

    @user_constraints.setter
    def user_constraints(self, constraints_dict: Dict[str, C]) -> None:
        self._constraints['user'] = constraints_dict

    def _quick_set(
        self,
        set_value: float,
        run_builtin_constraints: bool = False,
        run_user_constraints: bool = False,
        run_virtual_constraints: bool = False,
    ) -> None:
        """
        This is a quick setter for the parameter. It bypasses all the checks and constraints,
        just setting the value and issuing the interface callbacks.

        WARNING: This is a dangerous function and should only be used when you know what you are doing.
        """
        # First run the built-in constraints. i.e. min/max
        if run_builtin_constraints:
            constraint_type: MappingProxyType = self.builtin_constraints
            set_value = self.__constraint_runner(constraint_type, set_value)
        # Then run any user constraints.
        if run_user_constraints:
            constraint_type: dict = self.user_constraints
            state = self._global_object.stack.enabled
            if state:
                self._global_object.stack.force_state(False)
            try:
                set_value = self.__constraint_runner(constraint_type, set_value)
            finally:
                self._global_object.stack.force_state(state)
        if run_virtual_constraints:
            # And finally update any virtual constraints
            constraint_type: dict = self._constraints['virtual']
            _ = self.__constraint_runner(constraint_type, set_value)

        # Finally set the value
        self._property_value._magnitude._nominal_value = set_value
        self._args['value'] = set_value
        if self._callback.fset is not None:
            self._callback.fset(set_value)

    def __constraint_runner(
        self,
        this_constraint_type: Union[dict, MappingProxyType[str, C]],
        newer_value: numbers.Number,
    ) -> float:
        for constraint in this_constraint_type.values():
            if constraint.external:
                constraint()
                continue
            this_new_value = constraint(no_set=True)
            if this_new_value != newer_value:
                if global_object.debug:
                    print(f'Constraint `{constraint}` has been applied')
                self._value = self.__class__._constructor(
                    value=this_new_value,
                    units=self._args['units'],
                    error=self._args['error'],
                )
            newer_value = this_new_value
        return newer_value

    @property
    def bounds(self) -> Tuple[numbers.Number, numbers.Number]:
        """
        Get the bounds of the parameter.

        :return: Tuple of the parameters minimum and maximum values
        """
        return self._min, self._max

    @bounds.setter
    def bounds(self, new_bound: Union[Tuple[numbers.Number, numbers.Number], numbers.Number]) -> None:
        """
        Set the bounds of the parameter. *This will also enable the parameter*.

        :param new_bound: New bounds. This can be a tuple of (min, max) or a single number (min).
        For changing the max use (None, max_value).
        """
        # Macro checking and opening for undo/redo
        close_macro = False
        if self._global_object.stack.enabled:
            self._global_object.stack.beginMacro('Setting bounds')
            close_macro = True
        # Have we only been given a single number (MIN)?
        if isinstance(new_bound, numbers.Number):
            self.min = new_bound
        # Have we been given a tuple?
        if isinstance(new_bound, tuple):
            new_min, new_max = new_bound
            # Are there any None values?
            if isinstance(new_min, numbers.Number):
                self.min = new_min
            if isinstance(new_max, numbers.Number):
                self.max = new_max
        # Enable the parameter if needed
        if not self.enabled:
            self.enabled = True
        # This parameter is now not fixed.
        self.fixed = False
        # Close the macro if we opened it
        if close_macro:
            self._global_object.stack.endMacro()
