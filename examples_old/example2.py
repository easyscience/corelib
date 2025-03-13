__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

import numpy as np

from easyscience.fitting import Fitter
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.ObjectClasses import Parameter

# In this case we have inherited from `BaseObj` to create a class which has fitable attributes.


class Line(BaseObj):
    """
    Simple descriptor of a line.
    """

    _m = 1
    _c = 0

    def __init__(self):
        super().__init__(self.__class__.__name__, *self._defaults)

    @property
    def _defaults(self):
        return [Parameter("m", self._m), Parameter("c", self._c)]

    @property
    def gradient(self):
        return self.m.value

    @property
    def intercept(self):
        return self.c.value

    def fit_func(self, x: np.ndarray) -> np.ndarray:
        return self.gradient * x + self.intercept

    def __repr__(self):
        return f"Line: m={self.m}, c={self.c}"


l = Line()
f = Fitter()
f.initialize(l, l.fit_func)

x = np.array([1, 2, 3])
y = np.array([2, 4, 6]) - 1

f_res = f.fit(x, y)

print(f_res.chi2)
print(l)
