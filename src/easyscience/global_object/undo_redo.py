# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import abc
import functools
import logging
from collections import UserDict
from collections import deque
from typing import Any
from typing import Callable
from typing import Iterable
from typing import NoReturn
from typing import TypeVar
from typing import Union

import numpy as np

# Removed: from easyscience import global_object  # Avoid circular import


class UndoCommand(metaclass=abc.ABCMeta):
    """The Command interface pattern."""

    def __init__(self, obj) -> None:
        self._obj = obj
        self._text = None

    @abc.abstractmethod
    def undo(self) -> NoReturn:
        """Undo implementation which should be overwritten."""

    @abc.abstractmethod
    def redo(self) -> NoReturn:
        """Redo implementation which should be overwritten."""

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, text: str) -> NoReturn:
        self._text = text


T_ = TypeVar('T_', bound=UndoCommand)


def dict_stack_deco(func: Callable) -> Callable:

    def inner(obj, *args, **kwargs):
        from easyscience import global_object  # Local import to avoid circular dependency

        # Only do the work to a NotarizedDict.
        if hasattr(obj, '_stack_enabled') and obj._stack_enabled:
            if not kwargs:
                global_object.stack.push(DictStack(obj, *args))
            else:
                global_object.stack.push(DictStackReCreate(obj, **kwargs))
        else:
            func(obj, *args, **kwargs)

    return inner


class NotarizedDict(UserDict):
    """
    A simple dict drop in for EasyScience group classes.

    This is used as it wraps the get/set methods
    """

    def __init__(self, **kwargs):
        from easyscience import global_object  # Local import to avoid circular dependency

        super().__init__(**kwargs)
        self._global_object = global_object
        self._stack_enabled = False

    @classmethod
    def _classname(cls):
        # This method just returns the name of the class
        return cls.__name__

    @dict_stack_deco
    def __setitem__(self, key, value):
        super(NotarizedDict, self).__setitem__(key, value)

    @dict_stack_deco
    def __delitem__(self, key):
        super(NotarizedDict, self).__delitem__(key)

    def __repr__(self):
        return f'{self._classname()}({self.data})'

    @dict_stack_deco
    def reorder(self, **kwargs):
        self.data = kwargs.copy()


class CommandHolder:
    """
    A holder for one or more commands which are added to the stack.
    """

    def __init__(self, text: str = None):
        self._commands = deque()
        self._text = text
        self.__index = 0

    def append(self, command: T_):
        self._commands.appendleft(command)

    def pop(self):
        return self._commands.popleft()

    def __iter__(self) -> T_:
        while self.__index < len(self):
            index = self.__index
            self.__index += 1
            yield self._commands[index]
        self.__index = 0

    def __len__(self) -> int:
        return len(self._commands)

    @property
    def is_macro(self) -> bool:
        return len(self) > 1

    @property
    def current(self) -> T_:
        return self._commands[0]

    @property
    def text(self) -> str:
        text = ''
        if self._commands:
            text = self._commands[-1].text
        if self._text is not None:
            text = self._text
        return text

    @text.setter
    def text(self, text: str):
        self._text = text


class UndoStack:
    """Implement a version of QUndoStack without the QT."""

    def __init__(self, max_history: Union[int, type(None)] = None):
        self._history = deque(maxlen=max_history)
        self._future = deque(maxlen=max_history)
        self._macro_running = False
        self._command_running = False
        self._max_history = max_history
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, state: bool):
        if self.enabled and self._macro_running:
            self.endMacro()
        self._enabled = state

    def force_state(self, state: bool):
        self._enabled = state

    @property
    def history(self) -> deque:
        return self._history

    @property
    def future(self) -> deque:
        return self._future

    def push(self, command: T_) -> NoReturn:
        """Add a command to the history stack."""
        # If we're not enabled, then what are we doing!
        if not self.enabled or self._command_running:
            # Do the command and leave.
            command.redo()
            return
        # If there's a macro add the command to the command holder
        if self._macro_running:
            self.history[0].append(command)
        else:
            # Else create the command holder and add it to the stack
            com = CommandHolder()
            com.append(command)
            self.history.appendleft(com)
        # Actually do the command
        command.redo()
        # Reset the future
        self._future = deque(maxlen=self._max_history)

    def pop(self) -> T_:
        """
        !! WARNING - TO BE USED WITH EXTREME CAUTION !! !! THIS IS
        PROBABLY NOT THE FN YOU'RE LOOKING FOR, IT CAN BREAK A LOT OF
        STUFF !! Sometimes you really don't want the last command.
        Remove it from the stack

        Returns
        -------
        T_
            None.
        """
        pop_it = self._history.popleft()
        popped = pop_it.pop()
        if len(pop_it) > 0:
            self.history.appendleft(pop_it)
        return popped

    def clear(self) -> None:  # NoReturn:
        """Remove any commands on the stack and reset the state."""
        self._history = deque(maxlen=self._max_history)
        self._future = deque(maxlen=self._max_history)
        self._macro_running = False

    def undo(self) -> NoReturn:
        """Undo the last change to the stack."""
        if self.canUndo():
            # Move the command from the past to the future
            this_command_stack = self._history.popleft()
            self._future.appendleft(this_command_stack)

            # Execute all undo commands
            for command in this_command_stack:
                try:
                    self._command_running = True
                    command.undo()
                except Exception as e:
                    logging.getLogger('easyscience.global_object.undo_redo').warning(str(e))
                finally:
                    self._command_running = False

    def redo(self) -> NoReturn:
        """Redo the last ``undo`` command on the stack."""
        if self.canRedo():
            # Move from the future to the past
            this_command_stack = self._future.popleft()
            self._history.appendleft(this_command_stack)
            # Need to go from right to left
            this_command_stack = list(this_command_stack)
            this_command_stack.reverse()
            for command in this_command_stack:
                try:
                    self._command_running = True
                    command.redo()
                except Exception as e:
                    logging.getLogger('easyscience.global_object.undo_redo').warning(str(e))
                finally:
                    self._command_running = False

    def beginMacro(self, text: str) -> NoReturn:
        """
        Start a bulk update i.e. multiple commands under one undo/redo
        command.
        """
        if self._macro_running:
            raise AssertionError('Cannot start a macro when one is already running')
        com = CommandHolder(text)
        self.history.appendleft(com)
        self._macro_running = True

    def endMacro(self) -> NoReturn:
        """
        End a bulk update i.e. multiple commands under one undo/redo
        command.
        """
        if not self._macro_running:
            raise AssertionError('Cannot end a macro when one is not running')
        self._macro_running = False

    def canUndo(self) -> bool:
        """Can the last command be undone?"""
        return len(self._history) > 0 and not self._macro_running

    def canRedo(self) -> bool:
        """Can we redo a command?"""
        return len(self._future) > 0 and not self._macro_running

    def redoText(self) -> str:
        """Text associated with a redo item."""
        text = ''
        if self.canRedo():
            text = self.future[0].text
        return text

    def undoText(self) -> str:
        """Text associated with a undo item."""
        text = ''
        if self.canUndo():
            text = self.history[0].text
        return text


class PropertyStack(UndoCommand):
    """Stack operator for when a property setter is wrapped."""

    def __init__(self, parent, func: Callable, old_value: Any, new_value: Any, text: str = None):
        # self.setText("Setting {} to {}".format(func.__name__, new_value))
        super().__init__(self)
        self._parent = parent
        self._old_value = old_value
        self._new_value = new_value
        self._set_func = func
        self.text = f'{parent} value changed from {old_value} to {new_value}'
        if text is not None:
            self.text = text

    def undo(self) -> NoReturn:
        self._set_func(self._parent, self._old_value)

    def redo(self) -> NoReturn:
        self._set_func(self._parent, self._new_value)


class FunctionStack(UndoCommand):
    def __init__(self, parent, set_func: Callable, unset_func: Callable, text: str = None):
        super().__init__(self)
        self._parent = parent
        self._old_fn = set_func
        self._new_fn = unset_func
        self.text = f'{parent} called {set_func}'
        if text is not None:
            self.text = text

    def undo(self):
        self._new_fn()

    def redo(self):
        self._old_fn()


class DictStack(UndoCommand):
    def __init__(self, in_dict: NotarizedDict, *args):
        super().__init__(self)
        self._parent = in_dict

        self._deletion = False
        self._creation = False

        self._key = None
        self._index = None
        self._old_value = None
        self._new_value = None
        self.text = ''

        if len(args) == 1:
            # We are deleting
            self._deletion = True
            self._index = list(self._parent.keys()).index(args[0])
            self._old_value = self._parent[args[0]]
            self._key = args[0]
            self.text = f'Deleting {args[0]} from {self._parent}'
        elif len(args) == 2:
            # We are either creating or setting
            self._key = args[0]
            self._new_value = args[1]
            if self._key in self._parent.keys():
                # We are modifying
                self._old_value = self._parent[self._key]
                self.text = f'Setting {self._parent}[{self._key}] from {self._old_value} to {self._new_value}'
            else:
                self._creation = True
                self.text = f'Creating {self._parent}[{self._key}] with value {self._new_value}'
        else:
            raise ValueError

    def undo(self) -> NoReturn:
        if self._creation:
            # Now we delete
            self._parent.data.__delitem__(self._key)
        else:
            # Now we create/change value
            if self._index is None:
                self._parent.data.__setitem__(self._key, self._old_value)
            else:
                # This deals with placing an item in a place
                keys = list(self._parent.keys())
                values = list(self._parent.values())
                keys.insert(self._index, self._key)
                values.insert(self._index, self._old_value)
                self._parent.reorder(**{k: v for k, v in zip(keys, values)})

    def redo(self) -> NoReturn:
        if self._deletion:
            # Now we delete
            self._parent.data.__delitem__(self._key)
        else:
            self._parent.data.__setitem__(self._key, self._new_value)


class DictStackReCreate(UndoCommand):
    def __init__(self, in_dict: NotarizedDict, **kwargs):
        super().__init__(self)
        self._parent = in_dict
        self._old_value = in_dict.data.copy()
        self._new_value = kwargs
        self.text = 'Updating dictionary'

    def undo(self) -> NoReturn:
        self._parent.data = self._old_value

    def redo(self) -> NoReturn:
        self._parent.data = self._new_value


def property_stack(arg: Union[str, Callable], begin_macro: bool = False) -> Callable:
    """
    Decorate a ``property`` setter with undo/redo functionality This
    decorator can be used as:

    @property_stack def func() ....

    or

    @property_stack("This is the undo/redo text) def func() ....

    In the latter case the argument is a string which might be
    evaluated. The possible markups for this string are;

    ``obj`` - The thing being operated on ``func`` - The function being
    called ``name`` - The name of the function being called.
    ``old_value`` - The pre-set value ``new_value`` - The post-set value

    An example would be ``Function {name}: Set from {old_value} to
    {new_value}``
    """

    def make_wrapper(func: Callable, name: str, **kwargs) -> Callable:
        def wrapper(obj, *args) -> NoReturn:
            from easyscience import global_object  # Local import to avoid circular dependency

            old_value = getattr(obj, name)
            new_value = args[0]
            if issubclass(type(old_value), Iterable) or issubclass(type(new_value), Iterable):
                ret = np.all(old_value == new_value)
            else:
                ret = old_value == new_value
            if ret:
                return

            if global_object.debug:
                logging.getLogger('easyscience.global_object.undo_redo').debug(
                    "I'm %s and have been set from %s to %s!", obj, old_value, new_value
                )

            global_object.stack.push(PropertyStack(obj, func, old_value, new_value, **kwargs))

        return functools.update_wrapper(wrapper, func)

    if isinstance(arg, Callable):
        func = arg
        name = func.__name__
        wrapper = make_wrapper(func, name)
        setattr(wrapper, 'func', func)
    else:
        txt = arg

        def wrapper(func: Callable) -> Callable:
            name = func.__name__
            inner_wrapper = make_wrapper(func, name, text=txt.format(**locals()))
            setattr(inner_wrapper, 'func', func)
            return inner_wrapper

    return wrapper
