__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import time

import matplotlib.pyplot as plt

import numpy as np
from easyscience.Datasets.xarray import xr
from easyscience.fitting import Fitter
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.ObjectClasses import Parameter

d = xr.Dataset()

nx = 5E2
x_min = 0
x_max = 3

sin_offest = - 0.35
cos_offset = 0.14

x = np.linspace(x_min, x_max, num=int(nx))

d.easyscience.add_coordinate('x', x)
d.easyscience.add_coordinate('y', x + 1)
d.easyscience.add_variable('z', ['x', 'y'], np.sin(2*np.pi*(x + sin_offest)).reshape((-1, 1))*np.cos(2*np.pi*(x + cos_offset)).reshape((1, -1)) + 2*(0.5 - np.random.random(size=(int(nx), int(nx)))))

s_off_start_point = 0
c_off_start_point = 0

b = BaseObj('line',
            s_off=Parameter('s_off', s_off_start_point),
            c_off=Parameter('c_off', c_off_start_point))


def fit_fun(x, *args, **kwargs):
    # In the real case we would gust call the evaluation fn without reference to the BaseObj
    return np.sin(2*np.pi*(x[:, 0] + b.s_off.value)) * np.cos(2*np.pi*(x[:, 1] + b.c_off.value))


f = Fitter()
f.initialize(b, fit_fun)

print('Performing fit - No dask')
t = time.time()
f_res = d['z'].easyscience.fit(f)
d['computed_no_dask'] = f_res.y_calc
temp = d['z'] - d['computed_no_dask']
print(f'Time taken: {time.time() - t}')

print(f_res.p0)
print(f_res.p)

fig, axes = plt.subplots(ncols=2, nrows=2)
d['z'].plot(ax=axes[0, 0])
d['computed_no_dask'].plot(ax=axes[0, 1])
temp.plot(ax=axes[1, 0])
plt.tight_layout()
plt.show()



b.s_off.value = s_off_start_point
b.c_off.value = c_off_start_point

f = Fitter()
f.initialize(b, fit_fun)

print('Performing fit - Dask')
t = time.time()
to_chunk = ['z']
print('Chunking -->')
for name in to_chunk:
    d[name] = d[name].chunk({'x': 100, 'y': 100})
print('Fitting -->')
f_res = d['z'].easyscience.fit(f, dask='parallelized')
d['computed_dask'] = f_res.y_calc
temp = d['z'] - d['computed_dask']
print(f'Time taken: {time.time() - t}')

print(f_res.p0)
print(f_res.p)

fig, axes = plt.subplots(ncols=2, nrows=2)
d['z'].plot(ax=axes[0, 0])
d['computed_dask'].plot(ax=axes[0, 1])
temp.plot(ax=axes[1, 0])
plt.tight_layout()
plt.show()

print('All done :-)')
