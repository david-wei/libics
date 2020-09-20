from .base import *
from . import colors


import matplotlib.pyplot


def __getattr__(name):
    return getattr(matplotlib.pyplot, name)
