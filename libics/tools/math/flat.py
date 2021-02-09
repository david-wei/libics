import numpy as np

from libics.env import logging
from libics.tools.math.models import ModelBase


###############################################################################
# Oscillating Functions
###############################################################################


def cosine_2d(
    var, amplitude_x, amplitude_y, period_x, period_y, phase_x, phase_y,
    offset=0.0
):
    return (
        amplitude_x * np.cos(2 * np.pi * var[0] / period_x + phase_x)
        + amplitude_y * np.cos(2 * np.pi * var[1] / period_y + phase_y)
        + offset
    )


###############################################################################
# Monotonic Functions
###############################################################################


def linear_1d(x, a, c=0.0):
    r"""
    Linear in one dimension.

    .. math::
        a x + c
    """
    return a * x + c


class FitLinear1d(ModelBase):

    """
    Fit class for :py:func:`linear_1d`.

    Parameters
    ----------
    a (amplitude)
    c (offset)
    """

    LOGGER = logging.get_logger("libics.tools.math.flat.FitLinear1d")
    P_ALL = ["a", "c"]
    P_DEFAULT = [1, 0]

    @staticmethod
    def _func(var, *p):
        return linear_1d(var, *p)

    def find_p0(self, *data):
        var_data, func_data = self._split_fit_data(*data)
        var_data = var_data.ravel()
        idx_min, idx_max = np.argmin(func_data), np.argmax(func_data)
        fmin, fmax = func_data[idx_min], func_data[idx_max]
        vmin, vmax = var_data[idx_min], var_data[idx_max]
        a = (fmax - fmin) / (vmax - vmin)
        c = fmax - a * vmax
        self.p0 = [a, c]
