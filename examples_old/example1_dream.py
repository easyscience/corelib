__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

import numpy as np

from easyscience.fitting import Fitter
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.ObjectClasses import Parameter

# This is a simple example of creating an object which has fitable parameters

b = BaseObj("line", m=Parameter("m", 1), c=Parameter("c", 1))


def fit_fun(x):
    # In the real case we would gust call the evaluation fn without reference to the BaseObj
    return b.c.value + b.m.value * x


f = Fitter()
f.initialize(b, fit_fun)
f.switch_minimizer("bumps")

x = np.array([1, 2, 3])
y = np.array([2, 4, 6]) - 1

method = "dream"
dream_kwargs = {
    k: v
    for k, v in [
        ("samples", 10000),
        ("burn", 100),
        ("pop", 10),
        ("init", "eps"),
        ("thin", 1),
        ("alpha", 0.01),
        ("outliers", "none"),
        ("trim", False),
        ("steps", 0),
    ]
}

f_res = f.fit(x, y, method="dream", minimizer_kwargs=dream_kwargs)

print(f_res.chi2)
