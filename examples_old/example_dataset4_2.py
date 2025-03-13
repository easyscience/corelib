__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import matplotlib.pyplot as plt

import numpy as np
from easyscience.Datasets.xarray import xr
from easyscience.fitting import Fitter
from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.ObjectClasses import Parameter

d = xr.Dataset()


class Wavey(BaseObj):
    def __init__(self,
                 amplitude: Parameter,
                 phase: Parameter,
                 period: Parameter):
        super(Wavey, self).__init__('SimpleWave',
                                    amplitude=amplitude, phase=phase,
                                    period=period)

    def __repr__(self):
        return (f'{self.name} - A={self.amplitude}, ph={self.phase}, period={self.period}')

    @classmethod
    def from_params(cls, amplitude: float = 1, phase: float = 0, period: float = 2*np.pi):
        amplitude = Parameter('amplitude', amplitude, min=0)
        phase = Parameter('phase', phase)
        period = Parameter('period', period, min=0, max=2*np.pi)
        return cls(amplitude, phase, period)

    def fit_fun(self, x, *args, **kwargs):
        # In the real case we would gust call the evaluation fn without reference to the BaseObj
        return self.amplitude.value * np.sin((x + self.phase.value)/self.period.value)

b = Wavey.from_params()
bb = Wavey.from_params(1.1, 0.1, 1.9*np.pi)
f = Fitter()

f.initialize(b, b.fit_fun)

nx = 1E3
x_min = 0
x_max = 100

x = np.linspace(x_min, x_max, num=int(nx))
y1 = bb.fit_fun(x) + .75*(0.5 - np.random.random(size=x.shape))
x2 = x + 20
y2 = bb.fit_fun(x2) + .75*(0.5 - np.random.random(size=x2.shape))

d.easyscience.add_coordinate('x1', x)
d.easyscience.add_variable('y1', ['x1'], y1, auto_sigma=True)
d.easyscience.add_coordinate('x2', x2)
d.easyscience.add_variable('y2', ['x2'], y2, auto_sigma=True)

res = d.easyscience.fit(f, ['y1', 'y2'])

fig, axs = plt.subplots(1, len(res), sharey=True)
for idx, r in enumerate(res):
    r.y_obs.plot(ax=axs[idx])
    r.y_calc.plot(ax=axs[idx])
    axs[idx].set_title(f'Dataset {idx}')
plt.show()
print(f'Actual = {bb}')
print(f'Fitted = {b}')
