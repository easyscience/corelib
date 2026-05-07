# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import sys
from functools import wraps
from typing import Any
from typing import Callable
from typing import List

from easyscience import global_object

from .hugger import PatcherFactory
from .hugger import Store


class LoggedProperty(property):
    """
    Pump up python properties.

    In this case we can see who has called this property and then do
    something if a criteria is met. In this case if the caller is not a
    member of the ``ObjBase`` class. Note that all high level
    ``EasyScience`` objects should be built from ``ObjBase``.
    """

    _global_object = global_object

    def __init__(self, *args, get_id=None, my_self=None, test_class=None, **kwargs):
        super(LoggedProperty, self).__init__(*args, **kwargs)
        self._get_id = get_id
        self._my_self = my_self
        self.test_class = test_class

    @staticmethod
    def _caller_class(test_class, skip: int = 1):
        """Caller class."""

        def stack_(frame):
            """Stack function."""
            frame_list = []
            while frame:
                frame_list.append(frame)
                frame = frame.f_back
            return frame_list

        stack: List[Any] = stack_(sys._getframe(1))
        start = 0 + skip
        if len(stack) < start + 1:
            return ''
        parent_frame = stack[start]
        test = False
        if 'self' in parent_frame.f_locals:
            test = issubclass(parent_frame.f_locals['self'].__class__, test_class)
        return test

    def __get__(self, instance, owner=None):
        """Get function."""
        if not global_object.script.enabled:
            return super(LoggedProperty, self).__get__(instance, owner)
        test = self._caller_class(self.test_class)
        res = super(LoggedProperty, self).__get__(instance, owner)

        def result_item(item_to_be_resulted):
            """Result item."""
            if item_to_be_resulted is None:
                return None
            if global_object.map.is_known(item_to_be_resulted):
                global_object.map.change_type(item_to_be_resulted, 'returned')
            else:
                global_object.map.add_vertex(item_to_be_resulted, obj_type='returned')

        if not test and self._get_id is not None and self._my_self is not None:
            if not isinstance(res, list):
                result_item(res)
            else:
                for item in res:
                    result_item(item)
            Store().append_log(self.makeEntry('get', res))
            if global_object.debug:  # noqa: S1006
                print(f"I'm {self._my_self} and {self._get_id} has been called from the outside!")
        return res

    def __set__(self, instance, value):
        """Set function."""
        if not global_object.script.enabled:
            return super().__set__(instance, value)
        test = self._caller_class(self.test_class)
        if not test and self._get_id is not None and self._my_self is not None:
            Store().append_log(self.makeEntry('set', value))
            if global_object.debug:  # noqa: S1006
                print(
                    f"I'm {self._my_self} and {self._get_id} has been set to {value} from the outside!"
                )
        return super().__set__(instance, value)

    def makeEntry(self, log_type, returns, *args, **kwargs) -> str:
        """Makeentry function."""
        temp = ''
        if returns is None:
            returns = []
        if not isinstance(returns, list):
            returns = [returns]
        if log_type == 'get':
            for var in returns:
                if var.unique_name in global_object.map.returned_objs:
                    index = global_object.map.returned_objs.index(var.unique_name)
                    temp += f'{Store().var_ident}{index}, '
            if len(returns) > 0:
                temp = temp[:-2]
                temp += ' = '
            if self._my_self.unique_name in global_object.map.created_objs:
                # for edge in route[::-1]:
                index = global_object.map.created_objs.index(self._my_self.unique_name)
                temp += f'{self._my_self.__class__.__name__.lower()}_{index}.{self._get_id}'
            if self._my_self.unique_name in global_object.map.created_internal:
                # We now have to trace....
                route = global_object.map.reverse_route(self._my_self)  # noqa: F841
                index = global_object.map.created_internal.index(self._my_self.unique_name)
                temp += f'{self._my_self.__class__.__name__.lower()}_{index}.{self._get_id}'
        elif log_type == 'set':
            if self._my_self.unique_name in global_object.map.created_objs:
                index = global_object.map.created_objs.index(self._my_self.unique_name)
                temp += f'{self._my_self.__class__.__name__.lower()}_{index}.{self._get_id} = '
            args = args[1:]
            for var in args:
                if var.unique_name in global_object.map.argument_objs:
                    index = global_object.map.argument_objs.index(var.unique_name)
                    temp += f'{Store().var_ident}{index}'
                elif var.unique_name in global_object.map.returned_objs:
                    index = global_object.map.returned_objs.index(var.unique_name)
                    temp += f'{Store().var_ident}{index}'
                elif var.unique_name in global_object.map.created_objs:
                    index = global_object.map.created_objs.index(var.unique_name)
                    temp += f'{self._my_self.__class__.__name__.lower()}_{index}'
                else:
                    if isinstance(var, str):
                        var = '"' + var + '"'
                    temp += f'{var}'
        else:
            print(f'{log_type} is not implemented yet. Sorry')
        temp += '\n'
        return temp


class PropertyHugger(PatcherFactory):
    # Properties are immutable, so need to be set at the parent level. However unlike `FunctionHugger` we can't traverse
    # the stack to get the parent. So, it and it's name has to be set at initialization. Boo!

    _global_object = global_object

    def __init__(self, klass, prop_name):
        """Init function."""
        super().__init__()
        self.klass = klass
        if isinstance(prop_name, tuple):
            self.prop_name = prop_name[0]
            self.property = prop_name[1]
        else:
            self.prop_name = prop_name
            self.property = klass.__dict__.get(prop_name)
        self.__patch_ref = {
            'fget': {'old': self.property.fget, 'patcher': self.patch_get},
            'fset': {'old': self.property.fset, 'patcher': self.patch_set},
            'fdel': {'old': self.property.fdel, 'patcher': self.patch_del},
        }

    def patch(self):
        """Patch function."""
        option = {}
        for key, item in self.__patch_ref.items():
            func = getattr(self.property, key)
            if func is not None:
                if global_object.debug:
                    print(f'Patching property {self.klass.__name__}.{self.prop_name}')
                patch_function: Callable = item.get('patcher')
                new_func = patch_function(func)
                option[key] = new_func
        setattr(self.klass, self.prop_name, property(**option))

    def restore(self):
        """Restore function."""
        if global_object.debug:
            print(f'Restoring property {self.klass.__name__}.{self.prop_name}')
        setattr(self.klass, self.prop_name, self.property)

    def patch_get(self, func: Callable) -> Callable:
        """Patch get."""

        @wraps(func)
        def inner(*args, **kwargs):
            """Inner function."""
            if global_object.debug:
                print(
                    f'{self.klass.__name__}.{self.prop_name} has been called with {args[1:]}, {kwargs}'
                )
            res = func(*args, **kwargs)
            self._append_args(*args, **kwargs)
            self._append_result(res)
            self._append_log(self.makeEntry('get', res, *args, **kwargs))
            return res

        return inner

    def patch_set(self, func: Callable) -> Callable:
        """Patch set."""

        @wraps(func)
        def inner(*args, **kwargs):
            """Inner function."""
            if global_object.debug:
                print(
                    f'{self.klass.__name__}.{self.prop_name} has been set with {args[1:]}, {kwargs}'
                )
            self._append_args(*args, **kwargs)
            self._append_log(self.makeEntry('set', None, *args, **kwargs))
            return func(*args, **kwargs)

        return inner

    def patch_del(self, func: Callable) -> Callable:
        """Patch del."""

        @wraps(func)
        def inner(*args, **kwargs):
            """Inner function."""
            if global_object.debug:
                print(f'{self.klass.__name__}.{self.prop_name} has been deleted.')
            self._append_log(self.makeEntry('del', None, *args, **kwargs))
            return func(*args, **kwargs)

        return inner

    def makeEntry(self, log_type, returns, *args, **kwargs) -> str:
        """Makeentry function."""
        temp = ''
        if returns is None:
            returns = []
        if not isinstance(returns, list):
            returns = [returns]
        if log_type == 'get':
            for var in returns:
                if self._in_list('return_list', var):
                    index = self._get_position('return_list', var)
                    temp += f'{self._store.var_ident}{index}, '
            if len(returns) > 0:
                temp = temp[:-2]
                temp += ' = '
            if self._in_list('create_list', args[0]):
                index = self._get_position('create_list', args[0])
                temp += f'{self.klass.__name__.lower()}_{index}.{self.prop_name}'
        elif log_type == 'set':
            if self._in_list('create_list', args[0]):
                index = self._get_position('create_list', args[0])
                temp += f'{self.klass.__name__.lower()}_{index}.{self.prop_name} = '
            args = args[1:]
            for var in args:
                if self._in_list('input_list', var):
                    index = self._get_position('input_list', var)
                    temp += f'{self._store.var_ident}{index}'
                elif self._in_list('return_list', var):
                    index = self._get_position('return_list', var)
                    temp += f'{self._store.var_ident}{index}'
                elif self._in_list('create_list', var):
                    index = self._get_position('create_list', var)
                    temp += f'{self.klass.__name__.lower()}_{index}'
                else:
                    if isinstance(var, str):
                        var = '"' + var + '"'
                    temp += f'{var}'
        else:
            print(f'{log_type} is not implemented yet. Sorry')
        temp += '\n'
        return temp
