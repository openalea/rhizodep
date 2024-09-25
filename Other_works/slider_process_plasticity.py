"""
Simple script to check variations are coherent
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def MM(X, vmax, km):
    return X*vmax/(X+km)


def surfaces(X):
    return 2*X


# Richard's generalized logistic function
def Km(X, high, low, span, begin):
    precision = 0.99
    Q = precision/((1-precision)*np.exp(-begin))
    return (high-low)/(1+Q*np.exp(-X*(1/span)))+low


# The function to be called anytime a slider's value changes
def update(val):
    km = Km(Nm_slider.val, high_slider.val, low_slider.val, span_slider.val, begin_slider.val)
    line.set_ydata(MM(Nm_soil, vmax_slider.val, km))
    line2.set_ydata(Km(Nm, high_slider.val, low_slider.val, span_slider.val, begin_slider.val))
    ax[1].relim()
    ax[1].autoscale_view()
    fig.canvas.draw_idle()


Nm_soil = np.linspace(0, 1, 1000)
Nm = np.linspace(0, 1, 1000)


init_vmax = 1
init_Nm = 0.1
init_high = 0.95
init_low = 0.1
init_span = 0.03
init_begin = 6
init_km = Km(init_Nm, init_high, init_low, init_span, init_begin)


# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(2)
line, = ax[0].plot(Nm_soil, MM(Nm_soil, init_vmax, init_km))
line2, = ax[1].plot(Nm, Km(Nm, init_high, init_low, init_span, init_begin))
ax[0].set_title('MM kinetic = f(Nm_soil)')
ax[1].set_title('Km = f(Nm_root)')

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)

axvmax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
vmax_slider = Slider(
    ax=axvmax,
    label='Vmax [umol.s-1.m-2]',
    valmin=0,
    valmax=10,
    valinit=init_vmax,
)

axspan = fig.add_axes([0.25, 0.05, 0.65, 0.03])
span_slider = Slider(
    ax=axspan,
    label='Affinity transition span',
    valmin=0.001,
    valmax=1,
    valinit=init_span,
)

axbegin = fig.add_axes([0.25, 0.02, 0.65, 0.03])
begin_slider = Slider(
    ax=axbegin,
    label='Transition location',
    valmin=0,
    valmax=10,
    valinit=init_span,
)


axNm = fig.add_axes([0.05, 0.32, 0.0225, 0.63])
Nm_slider = Slider(
    ax=axNm,
    label="Nm_root [mol.g-1]",
    valmin=0,
    valmax=1,
    valinit=init_Nm,
    orientation='vertical'
)

axlow = fig.add_axes([0.1, 0.27, 0.0225, 0.63])
low_slider = Slider(
    ax=axlow,
    label="HATS [mol.g-1]",
    valmin=0,
    valmax=1,
    valinit=init_low,
    orientation='vertical'
)

axhigh = fig.add_axes([0.15, 0.22, 0.0225, 0.63])
high_slider = Slider(
    ax=axhigh,
    label='LATS [mol.g-1]',
    valmin=0,
    valmax=1,
    valinit=init_high,
    orientation='vertical'
)


# register the update function with each slider
vmax_slider.on_changed(update)
span_slider.on_changed(update)
begin_slider.on_changed(update)
Nm_slider.on_changed(update)
high_slider.on_changed(update)
low_slider.on_changed(update)

plt.show()