__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

from typing import Callable

import numpy as np

from easyscience.fitting import Fitter
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.ObjectClasses import Parameter

# In this case we have inherited from `BaseObj` to create a class which has fitable attributes.
# This class does not know about the `Calculator`, only the interface.


class Calculator:
    def __init__(self, m=1, c=0):
        self.m = m
        self.c = c

    def calculate(self, x):
        return self.m * x + self.c


class Interface:
    def __init__(self):
        self.calculator = Calculator()

    def get_value(self, value_label: str):
        return getattr(self.calculator, value_label, None)

    def set_value(self, value_label: str, value):
        print(f'Value of {value_label} set to {value}')
        setattr(self.calculator, value_label, value)

    def fit_func(self, x):
        return self.calculator.calculate(x)


class Line(BaseObj):
    _defaults = [Parameter('m', 1),
                 Parameter('c', 0)]

    def __init__(self, interface=None):
        super().__init__(self.__class__.__name__,
                         *self._defaults)
        self.interface = interface
        if self.interface:
            for parameter in self.get_fit_parameters():
                name = parameter.name
                setattr(self.__class__.__dict__[name],
                        '_callback',
                        property(self.__gitem(name), self.__sitem(self, name)))

    @property
    def gradient(self):
        if self.interface:
            return self.interface.get_value('m')
        else:
            return self.m.value

    @property
    def intercept(self):
        if self.interface:
            return self.interface.get_value('c')
        else:
            return self.c.value

    def fit_func(self, x: np.ndarray) -> np.ndarray:
        if self.interface:
            return self.interface.fit_func(x)
        else:
            raise NotImplementedError

    def __repr__(self):
        return f'Line: m={self.m}, c={self.c}'

    @staticmethod
    def __gitem(key: str) -> Callable:
        def inner(obj):
            obj.interface.get_value(key)
        return lambda obj: inner(obj)

    @staticmethod
    def __sitem(obj, key):
        def inner(value):
            obj.interface.set_value(key, value)
        return inner


l = Line(interface=Interface())
l.interface
f = Fitter(l, l.fit_func)

x = np.array([1, 2, 3])
y = np.array([2, 4, 6]) - 1

f_res = f.fit(x, y)

print(f_res)
print(l)