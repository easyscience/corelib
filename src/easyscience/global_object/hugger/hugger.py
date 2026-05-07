# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import sys
from abc import ABCMeta
from abc import abstractmethod
from typing import List
from typing import Tuple

from ...utils.classUtils import singleton


@singleton
class Store:
    __log = []
    __var_ident = 'var_'
    __ret_ident = 'ret_'

    def __init__(self):
        """Init function."""
        self.log = self.__log  # TODO Async problem?
        self.var_ident = self.__var_ident
        self.ret_ident = self.__ret_ident

    @staticmethod
    def get_defaults() -> dict:
        """Get defaults."""
        return {
            'log': Store.__log,
            'create_list': Store.__create_list,
            'unique_args': Store.__unique_args,
            'unique_rets': Store.__unique_rets,
            'var_ident': Store.__var_ident,
            'ret_ident': Store.__ret_ident,
        }

    def append_log(self, log_entry: str):
        """Append log."""
        self.log.append(log_entry)


class ScriptManager:
    def __init__(self, enabled=True):
        """Init function."""
        self._store = Store()
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Enabled function."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Enabled function."""
        self._enabled = value

    def history(self) -> List[str]:
        """History function."""
        return self._store.log

    def reset_history(self):
        """Reset history."""
        defaults = Store.get_defaults()
        for key, item in defaults.items():
            setattr(self._store, key, item)

    def append_log(self, log_entry: str):
        """Append log."""
        self._store.log.append(log_entry)


class Hugger(metaclass=ABCMeta):
    def __init__(self):
        """Init function."""
        self._store = Store()

    @property
    def log(self) -> list:
        """Log function."""
        return self._store.log

    @abstractmethod
    def patch(self):
        """Patch function."""
        pass

    @abstractmethod
    def restore(self):
        """Restore function."""
        pass


class PatcherFactory(Hugger, metaclass=ABCMeta):
    def __init__(self):
        """Init function."""
        super().__init__()

    @staticmethod
    def is_mutable(arg) -> bool:
        """Is mutable."""
        ret = True
        if isinstance(arg, (int, float, complex, str, tuple, frozenset, bytes, property)):
            ret = False
        return ret

    @staticmethod
    def _caller_name(skip: int = 2):
        """
        Get a name of a caller in the format module.class.method
        ``skip`` specifies how many levels of stack to skip while
        getting caller name.

        skip=1 means "who calls me", skip=2 "who calls my caller" etc.
        An empty string is returned if skipped levels exceed stack
        height
        https://gist.github.com/techtonik/2151727#gistcomment-2333747
        """

        def stack_(frame):
            """Stack function."""
            framelist = []
            while frame:
                framelist.append(frame)
                frame = frame.f_back
            return framelist

        stack = stack_(sys._getframe(1))
        start = 0 + skip
        if len(stack) < start + 1:
            return ''
        parentframe = stack[start]

        name = []
        module = inspect.getmodule(parentframe)
        # `modname` can be None when frame is executed directly in console
        # TODO(techtonik): consider using __main__
        if module:
            name.append(module.__name__)
        # detect classname
        if 'self' in parentframe.f_locals:
            # I don't know any way to detect call from the object method
            # XXX: there seems to be no way to detect static method call - it will
            #      be just a function call
            name.append(parentframe.f_locals['self'].__class__.__name__)
        codename = parentframe.f_code.co_name
        if codename != '<module>':  # top level usually
            name.append(codename)  # function or a method
        del parentframe
        return '.'.join(name)

    def _append_args(self, *args, **kwargs):
        """Append args."""

        def check(res):
            """Check function."""
            return (
                id(res) not in self._store.unique_rets
                and id(res) not in self._store.create_list
                and id(res) not in self._store.unique_args
            )

        for arg in args:
            if self.is_mutable(arg) and check(arg):
                self._store.unique_args.append(id(arg))
        for item in kwargs.values():
            if self.is_mutable(item) and check(item):
                self._store.unique_args.append(id(item))

    def _append_create(self, obj):
        """Append create."""
        this_id = id(obj)
        if this_id not in self._store.create_list:
            self._store.create_list.append(this_id)

    def _append_result(self, result) -> int:
        """Append result."""
        ret = 0

        def check(res):
            """Check function."""
            return (
                id(res) not in self._store.unique_rets
                and id(res) not in self._store.create_list
                and id(res) not in self._store.unique_args
            )

        if isinstance(result, type(None)):
            return ret
        elif isinstance(result, tuple):
            for res in result:
                # if self.is_mutable(res) and check(res):
                if check(res):
                    self._store.unique_rets.append(id(res))
            ret = len(result)
        else:
            # if self.is_mutable(result) and check(result):
            if check(result):
                self._store.unique_rets.append(id(result))
            ret = 1
        return ret

    def _append_log(self, log_entry: str):
        """Append log."""
        self._store.log.append(log_entry)

    def __options(self, item) -> Tuple[int, dict]:
        """Options function."""
        this_id = id(item)
        option = {
            'create_list': self._store.create_list,
            'return_list': self._store.unique_rets,
            'input_list': self._store.unique_args,
        }
        return this_id, option

    def _get_position(self, query: str, item) -> int:
        """Get position."""
        this_id, option = self.__options(item)
        in_list = self._in_list(query, item)
        index = None
        if in_list:
            index = option.get(query).index(this_id)
        return index

    def _in_list(self, query: str, item) -> bool:
        """In list."""
        this_id, option = self.__options(item)
        return this_id in option.get(query, [])

    @staticmethod
    def _get_class_that_defined_method(method_in) -> classmethod:
        """Get class that defined method."""
        if inspect.ismethod(method_in):
            for cls in inspect.getmro(method_in.__self__.__class__):
                if cls.__dict__.get(method_in.__name__) is method_in:
                    return cls
            method_in = method_in.__func__  # fallback to __qualname__ parsing
        if inspect.isfunction(method_in):
            class_name = method_in.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0]
            try:
                cls = getattr(inspect.getmodule(method_in), class_name)
            except AttributeError:
                cls = method_in.__globals__.get(class_name)
            if isinstance(cls, type):
                return cls
