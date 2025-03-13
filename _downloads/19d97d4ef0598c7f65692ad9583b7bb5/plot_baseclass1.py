"""
Subclassing BaseObj - Simple Pendulum
=====================================
This  example shows how to subclass :class:`easyscience.Objects.Base.BaseObj` with parameters from
:class:`EasyScience.Objects.Base.Parameter`. For this example a simple pendulum will be modeled.

.. math::
    y = A \sin (2 \pi f t + \phi )

Imports
*******

Firstly the necessary imports. Notice that we import numpy from easyscience. This is not done for any reason other than
saving time from multiple imports.
"""

import matplotlib.pyplot as plt
import numpy as np

from easyscience.Objects.ObjectClasses import BaseObj
from easyscience.Objects.ObjectClasses import Parameter

# %%
# Subclassing
# ***********
# To include embedded rST, use a line of >= 20 ``#``'s or ``#%%`` between your
# rST and your code. This separates your example
# into distinct text and code blocks. You can continue writing code below the
# embedded rST text block:


class Pendulum(BaseObj):
    def __init__(self, A: Parameter, f: Parameter, p: Parameter):
        super(Pendulum, self).__init__('SimplePendulum', A=A, f=f, p=p)

    @classmethod
    def from_pars(cls, A: float = 1, f: float = 1, p: float = 0):
        A = Parameter('Amplitude', A)
        f = Parameter('Frequency', f)
        p = Parameter('Phase', p)
        return cls(A, f, p)

    def __call__(self, t):
        return self.A.value * np.sin(2 * np.pi * self.f.value * t + self.p.value)

    def plot(self, time, axis=None, **kwargs):
        if axis is None:
            axis = plt
        else:
            axis.set_title(f'A={self.A.value}, F={self.f.value}, P={self.p.value}')
        p = axis.plot(time, self(time), **kwargs)
        return p


# %%
# Single Example
# **************
# To include embedded rST, use a line of >= 20 ``#``'s or ``#%%`` between your
# rST and your code. This separates your example
# into distinct text and code blocks. You can continue writing code below the
# embedded rST text block:

p1 = Pendulum.from_pars()
# Another pendulum with Amplitude = 5
p2 = Pendulum.from_pars(A=5)
# Another pendulum with Frequency = 4
p3 = Pendulum.from_pars(A=5, f=4)
# Another pendulum with Phase = pi/2
p4 = Pendulum.from_pars(A=5, f=4, p=np.pi / 2)

# %%
# Plotting

t = np.linspace(0, 3, 601)
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
p1.plot(t, axis=ax1)
p2.plot(t, axis=ax2)
p3.plot(t, axis=ax3)
p4.plot(t, axis=ax4)
fig.show()

# %%
# Multiple Examples
# *****************
# To include embedded rST, use a line of >= 20 ``#``'s or ``#%%`` between your
# rST and your code. This separates your example
# into distinct text and code blocks. You can continue writing code below the
# embedded rST text block:

pendulum_array = [Pendulum.from_pars(p=phase) for phase in np.linspace(0, 1, 3)]
fig = plt.figure()
for pendulum in pendulum_array:
    pendulum.plot(t, label=f'Phase = {pendulum.p}')
plt.legend(loc='lower right')
fig.show()
