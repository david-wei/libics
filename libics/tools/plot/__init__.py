from .base import *
from . import colors
from . import default


import matplotlib.pyplot


def __getattr__(name):
    return getattr(matplotlib.pyplot, name)
