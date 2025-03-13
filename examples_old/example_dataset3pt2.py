__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

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


fig, ax = plt.subplots(2, 3, sharey=True, sharex=True)
cbar_ax1 = fig.add_axes([0.85, 0.15, 0.05, 0.3])
cbar_ax2 = fig.add_axes([0.85, 0.60, 0.05, 0.3])

for idx, minimizer in enumerate(['lmfit', 'bumps', 'dfo_ls']):
    b.s_off = s_off_start_point
    b.c_off = c_off_start_point

    f = Fitter()
    f.initialize(b, fit_fun)

    f_res = d['z'].easyscience.fit(f)
    d[f'computed_{minimizer}'] = f_res.y_calc
    d[f'dz_{minimizer}'] = d['z'] - d[f'computed_{minimizer}']

    p1 = d[f'computed_{minimizer}'].plot(ax=ax[0, idx], cbar_kwargs={'cax': cbar_ax1})
    p2 = d[f'dz_{minimizer}'].plot(ax=ax[1, idx], cbar_kwargs={'cax': cbar_ax2})
    ax[0, idx].set_title(f'{minimizer}')
    ax[1, idx].set_title('s_off - {:0.03f}\nc_off - {:0.03f}'.format(b.s_off.value, b.c_off.value))
    ax[0, idx].set_aspect('equal', 'box')
    ax[1, idx].set_aspect('equal', 'box')
fig.subplots_adjust(right=0.8)
plt.show()
