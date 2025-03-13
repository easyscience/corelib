__author__ = "github.com/wardsimon"
__version__ = "0.1.0"

import matplotlib.pyplot as plt

import numpy as np
from easyscience.Datasets.xarray import xr
from easyscience.fitting import Fitter
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.ObjectClasses import Parameter

d = xr.Dataset()

b = BaseObj("line", m=Parameter("m", 1), c=Parameter("c", 1))


def fit_fun(x, *args, **kwargs):
    # In the real case we would gust call the evaluation fn without reference to the BaseObj
    return b.c.value + b.m.value * x


f = Fitter()
f.initialize(b, fit_fun)

nx = 1e3
x_min = 0
x_max = 100

x = np.linspace(x_min, x_max, num=int(nx))
y = 2 * x - 1 + 5 * (np.random.random(size=x.shape) - 0.5)

d.easyscience.add_coordinate("x", x)
d.easyscience.add_variable("y", ["x"], y, auto_sigma=True)


def post(result, addition=10):
    return result + addition


d["y"].easyscience.postcompute_func = post

# d['y'] = d['y'].chunk({'x': 1000})
f_res = d["y"].easyscience.fit(f, dask="parallelized")

print(f_res.chi2)

d["y"].plot()
d["computed"] = f_res.y_calc
d["computed"].plot()
plt.show()
