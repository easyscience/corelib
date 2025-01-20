#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

from __future__ import annotations

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import weakref
from abc import ABCMeta
from abc import abstractmethod
from numbers import Number
from typing import TYPE_CHECKING
from typing import Callable
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy as np
from asteval import Interpreter

from easyscience import global_object
from easyscience.Objects.core import ComponentSerializer

if TYPE_CHECKING:
    from easyscience.Objects.Variable import V


class ConstraintBase(ComponentSerializer, metaclass=ABCMeta):
    """
    A base class used to describe a constraint to be applied to EasyScience base objects.
    """

    _global_object = global_object

    def __init__(
        self,
        dependent_obj: V,
        independent_obj: Optional[Union[V, List[V]]] = None,
        operator: Optional[Union[str, List[str]]] = None,
        value: Optional[Number] = None,
    ):
        self.aeval = Interpreter()
        self.dependent_obj_ids = dependent_obj.unique_name
        self.independent_obj_ids = None
        self._enabled = True
        self.external = False
        self._finalizer = None
        if independent_obj is not None:
            if isinstance(independent_obj, list):
                self.independent_obj_ids = [obj.unique_name for obj in independent_obj]
                if self.dependent_obj_ids in self.independent_obj_ids:
                    raise AttributeError('A dependent object can not be an independent object')
            else:
                self.independent_obj_ids = independent_obj.unique_name
                if self.dependent_obj_ids == self.independent_obj_ids:
                    raise AttributeError('A dependent object can not be an independent object')
            # Test if dependent is a parameter or a descriptor.
            # We can not import `Parameter`, so......
            if dependent_obj.__class__.__name__ == 'Parameter':
                if not dependent_obj.enabled:
                    raise AssertionError('A dependent object needs to be initially enabled.')
                if global_object.debug:
                    print(f'Dependent variable {dependent_obj}. It should be a `Descriptor`.' f'Setting to fixed')
                dependent_obj.enabled = False
                self._finalizer = weakref.finalize(self, cleanup_constraint, self.dependent_obj_ids, True)

        self.operator = operator
        self.value = value

    @property
    def enabled(self) -> bool:
        """
        Is the current constraint enabled.

        :return: Logical answer to if the constraint is enabled.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, enabled_value: bool):
        """
                Set the enabled state of the constraint. If the new value is the same as the current value only the state is
                changed.

        ... note:: If the new value is ``True`` the constraint is also applied after enabling.

                :param enabled_value: New state of the constraint.
                :return: None
        """

        if self._enabled == enabled_value:
            return
        elif enabled_value:
            self.get_obj(self.dependent_obj_ids).enabled = False
            self()
        else:
            self.get_obj(self.dependent_obj_ids).enabled = True
        self._enabled = enabled_value

    def __call__(self, *args, no_set: bool = False, **kwargs):
        """
        Method which applies the constraint

        :return: None if `no_set` is False, float otherwise.
        """
        if not self.enabled:
            if no_set:
                return None
            return
        independent_objs = None
        if isinstance(self.dependent_obj_ids, str):
            dependent_obj = self.get_obj(self.dependent_obj_ids)
        else:
            raise AttributeError
        if isinstance(self.independent_obj_ids, str):
            independent_objs = self.get_obj(self.independent_obj_ids)
        elif isinstance(self.independent_obj_ids, list):
            independent_objs = [self.get_obj(obj_id) for obj_id in self.independent_obj_ids]
        if independent_objs is not None:
            value = self._parse_operator(independent_objs, *args, **kwargs)
        else:
            value = self._parse_operator(dependent_obj, *args, **kwargs)

        if not no_set:
            toggle = False
            if not dependent_obj.enabled:
                dependent_obj.enabled = True
                toggle = True
            dependent_obj.value = value
            if toggle:
                dependent_obj.enabled = False
        return value

    @abstractmethod
    def _parse_operator(self, obj: V, *args, **kwargs) -> Number:
        """
        Abstract method which contains the constraint logic

        :param obj: The object/objects which the constraint will use
        :return: A numeric result of the constraint logic
        """

    @abstractmethod
    def __repr__(self):
        pass

    def get_obj(self, key: int) -> V:
        """
        Get an EasyScience object from its unique key

        :param key: an EasyScience objects unique key
        :return: EasyScience object
        """
        return self._global_object.map.get_item_by_key(key)


C = TypeVar('C', bound=ConstraintBase)


class NumericConstraint(ConstraintBase):
    """
    A `NumericConstraint` is a constraint whereby a dependent parameters value is something of an independent parameters
    value. I.e. a < 1, a > 5
    """

    def __init__(self, dependent_obj: V, operator: str, value: Number):
        """
        A `NumericConstraint` is a constraint whereby a dependent parameters value is something of an independent
        parameters value. I.e. a < 1, a > 5

        :param dependent_obj: Dependent Parameter
        :param operator: Relation to between the parameter and the values. e.g. ``=``, ``<``, ``>``
        :param value: What the parameters value should be compared against.

        :example:

        .. code-block:: python

            from easyscience.fitting.Constraints import NumericConstraint
            from easyscience.Objects.Base import Parameter
            # Create an `a < 1` constraint
            a = Parameter('a', 0.2)
            constraint = NumericConstraint(a, '<=', 1)
            a.user_constraints['LEQ_1'] = constraint
            # This works
            a.value = 0.85
            # This triggers the constraint
            a.value = 2.0
            # `a` is set to the maximum of the constraint (`a = 1`)
        """
        super(NumericConstraint, self).__init__(dependent_obj, operator=operator, value=value)

    def _parse_operator(self, obj: V, *args, **kwargs) -> Number:
        ## TODO Probably needs to be updated when DescriptorArray is implemented

        value = obj.value_no_call_back

        if isinstance(value, list):
            value = np.array(value)
        self.aeval.symtable['value1'] = value
        self.aeval.symtable['value2'] = self.value
        try:
            self.aeval.eval(f'value3 = value1 {self.operator} value2')
            logic = self.aeval.symtable['value3']
            if isinstance(logic, np.ndarray):
                value[not logic] = self.aeval.symtable['value2']
            else:
                if not logic:
                    value = self.aeval.symtable['value2']
        except Exception as e:
            raise e
        finally:
            self.aeval = Interpreter()
        return value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} with `value` {self.operator} {self.value}'


class SelfConstraint(ConstraintBase):
    """
    A `SelfConstraint` is a constraint which tests a logical constraint on a property of itself, similar to a
    `NumericConstraint`. i.e. a > a.min. These constraints are usually used in the internal EasyScience logic.
    """

    def __init__(self, dependent_obj: V, operator: str, value: str):
        """
        A `SelfConstraint` is a constraint which tests a logical constraint on a property of itself, similar to
        a `NumericConstraint`. i.e. a > a.min.

        :param dependent_obj: Dependent Parameter
        :param operator: Relation to between the parameter and the values. e.g. ``=``, ``<``, ``>``
        :param value: Name of attribute to be compared against

        :example:

        .. code-block:: python

            from easyscience.fitting.Constraints import SelfConstraint
            from easyscience.Objects.Base import Parameter
            # Create an `a < a.max` constraint
            a = Parameter('a', 0.2, max=1)
            constraint = SelfConstraint(a, '<=', 'max')
            a.user_constraints['MAX'] = constraint
            # This works
            a.value = 0.85
            # This triggers the constraint
            a.value = 2.0
            # `a` is set to the maximum of the constraint (`a = 1`)
        """
        super(SelfConstraint, self).__init__(dependent_obj, operator=operator, value=value)

    def _parse_operator(self, obj: V, *args, **kwargs) -> Number:
        value = obj.value_no_call_back

        self.aeval.symtable['value1'] = value
        self.aeval.symtable['value2'] = getattr(obj, self.value)
        try:
            self.aeval.eval(f'value3 = value1 {self.operator} value2')
            logic = self.aeval.symtable['value3']
            if isinstance(logic, np.ndarray):
                value[not logic] = self.aeval.symtable['value2']
            else:
                if not logic:
                    value = self.aeval.symtable['value2']
        except Exception as e:
            raise e
        finally:
            self.aeval = Interpreter()
        return value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} with `value` {self.operator} obj.{self.value}'


class ObjConstraint(ConstraintBase):
    """
    A `ObjConstraint` is a constraint whereby a dependent parameter is something of an independent parameter
    value. E.g. a (Dependent Parameter) = 2* b (Independent Parameter)
    """

    def __init__(self, dependent_obj: V, operator: str, independent_obj: V):
        """
        A `ObjConstraint` is a constraint whereby a dependent parameter is something of an independent parameter
        value. E.g. a (Dependent Parameter) < b (Independent Parameter)

        :param dependent_obj: Dependent Parameter
        :param operator: Relation to between the independent parameter and dependent parameter. e.g. ``2 *``, ``1 +``
        :param independent_obj: Independent Parameter

        :example:

        .. code-block:: python

            from easyscience.fitting.Constraints import ObjConstraint
            from easyscience.Objects.Base import Parameter
            # Create an `a = 2 * b` constraint
            a = Parameter('a', 0.2)
            b = Parameter('b', 1)

            constraint = ObjConstraint(a, '2*', b)
            b.user_constraints['SET_A'] = constraint
            b.value = 1
            # This triggers the constraint
            a.value # Should equal 2

        """
        super(ObjConstraint, self).__init__(dependent_obj, independent_obj=independent_obj, operator=operator)
        self.external = True

    def _parse_operator(self, obj: V, *args, **kwargs) -> Number:
        value = obj.value_no_call_back

        self.aeval.symtable['value1'] = value
        try:
            self.aeval.eval(f'value2 = {self.operator} value1')
            value = self.aeval.symtable['value2']
        except Exception as e:
            raise e
        finally:
            self.aeval = Interpreter()
        return value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} with `dependent_obj` = {self.operator} `independent_obj`'


class MultiObjConstraint(ConstraintBase):
    """
    A `MultiObjConstraint` is similar to :class:`EasyScience.fitting.Constraints.ObjConstraint` except that it relates to
    multiple independent objects.
    """

    def __init__(
        self,
        independent_objs: List[V],
        operator: List[str],
        dependent_obj: V,
        value: Number,
    ):
        """
        A `MultiObjConstraint` is similar to :class:`EasyScience.fitting.Constraints.ObjConstraint` except that it relates
        to one or more independent objects.

        E.g.
        * a (Dependent Parameter) + b (Independent Parameter) = 1
        * a (Dependent Parameter) + b (Independent Parameter) - 2*c (Independent Parameter) = 0

        :param independent_objs: List of Independent Parameters
        :param operator: List of operators operating on the Independent Parameters
        :param dependent_obj: Dependent Parameter
        :param value: Value of the expression

        :example:

        **a + b = 1**

        .. code-block:: python

            from easyscience.fitting.Constraints import MultiObjConstraint
            from easyscience.Objects.Base import Parameter
            # Create an `a + b = 1` constraint
            a = Parameter('a', 0.2)
            b = Parameter('b', 0.3)

            constraint = MultiObjConstraint([b], ['+'], a, 1)
            b.user_constraints['SET_A'] = constraint
            b.value = 0.4
            # This triggers the constraint
            a.value # Should equal 0.6

        **a + b - 2c = 0**

        .. code-block:: python

            from easyscience.fitting.Constraints import MultiObjConstraint
            from easyscience.Objects.Base import Parameter
            # Create an `a + b - 2c = 0` constraint
            a = Parameter('a', 0.5)
            b = Parameter('b', 0.3)
            c = Parameter('c', 0.1)

            constraint = MultiObjConstraint([b, c], ['+', '-2*'], a, 0)
            b.user_constraints['SET_A'] = constraint
            c.user_constraints['SET_A'] = constraint
            b.value = 0.4
            # This triggers the constraint. Or it could be triggered by changing the value of c
            a.value # Should equal 0.2

        .. note:: This constraint is evaluated as ``dependent`` = ``value`` - SUM(``operator_i`` ``independent_i``)
        """
        super(MultiObjConstraint, self).__init__(
            dependent_obj,
            independent_obj=independent_objs,
            operator=operator,
            value=value,
        )
        self.external = True

    def _parse_operator(self, independent_objs: List[V], *args, **kwargs) -> Number:

        in_str = ''
        value = None
        for idx, obj in enumerate(independent_objs):
            self.aeval.symtable['p' + str(self.independent_obj_ids[idx])] = obj.value_no_call_back

            in_str += ' p' + str(self.independent_obj_ids[idx])
            if idx < len(self.operator):
                in_str += ' ' + self.operator[idx]
        try:
            self.aeval.eval(f'final_value = {self.value} - ({in_str})')
            value = self.aeval.symtable['final_value']
        except Exception as e:
            raise e
        finally:
            self.aeval = Interpreter()
        return value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class FunctionalConstraint(ConstraintBase):
    """
    Functional constraints do not depend on other parameters and as such can be more complex.
    """

    def __init__(
        self,
        dependent_obj: V,
        func: Callable,
        independent_objs: Optional[List[V]] = None,
    ):
        """
        Functional constraints do not depend on other parameters and as such can be more complex.

        :param dependent_obj: Dependent Parameter
        :param func: Function to be evaluated in the form ``f(value, *args, **kwargs)``

        :example:

        .. code-block:: python

            import numpy as np
            from easyscience.fitting.Constraints import FunctionalConstraint
            from easyscience.Objects.Base import Parameter

            a = Parameter('a', 0.2, max=1)
            constraint = FunctionalConstraint(a, np.abs)

            a.user_constraints['abs'] = constraint

            # This triggers the constraint
            a.value = 0.85 # `a` is set to 0.85
            # This triggers the constraint
            a.value = -0.5 # `a` is set to 0.5
        """
        super(FunctionalConstraint, self).__init__(dependent_obj, independent_obj=independent_objs)
        self.function = func
        if independent_objs is not None:
            self.external = True

    def _parse_operator(self, obj: V, *args, **kwargs) -> Number:

        self.aeval.symtable[f'f{id(self.function)}'] = self.function
        value_str = f'r_value = f{id(self.function)}('
        if isinstance(obj, list):
            for o in obj:
                value_str += f'{o.value_no_call_back},'

            value_str = value_str[:-1]
        else:
            value_str += f'{obj.value_no_call_back}'

        value_str += ')'
        try:
            self.aeval.eval(value_str)
            value = self.aeval.symtable['r_value']
        except Exception as e:
            raise e
        finally:
            self.aeval = Interpreter()
        return value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


def cleanup_constraint(obj_id: str, enabled: bool):
    try:
        obj = global_object.map.get_item_by_key(obj_id)
        obj.enabled = enabled
    except ValueError:
        if global_object.debug:
            print(f'Object with ID {obj_id} has already been deleted')
