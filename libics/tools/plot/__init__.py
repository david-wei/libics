from .base import *
from . import colors
from .layout import SubfigLayout, SubfigMargins, SubfigSize, make_fixed_axes
from .multi import plot_ax_array


import matplotlib.pyplot


def __getattr__(name):
    return getattr(matplotlib.pyplot, name)
