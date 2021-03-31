import numpy as np
from numpy.core.fromnumeric import ndim
import scipy

from libics.env import logging
from libics.tools.math.models import ModelBase


###############################################################################
# Oscillating Functions
###############################################################################


def cosine_1d( var, amplitude, frequency, phase, offset=0.0):
    return amplitude * np.cos(2 * np.pi * frequency * var + phase) + offset


class FitCosine1d(ModelBase):

    """
    Fit class for :py:func:`cosine_1d`.

    Parameters
    ----------
    a (amplitude)
    f (frequency without 2π)
    phi (additive phase)
    c (offset)
    """

    LOGGER = logging.get_logger("libics.tools.math.flat.FitLinear1d")
    P_ALL = ["a", "f", "phi", "c"]
    P_DEFAULT = [1, 1, 0, 0]

    @staticmethod
    def _func(var, *p):
        return cosine_1d(var, *p)

    def find_p0(self, *data, MAX_POINTS=1024):
        var_data, func_data, _ = self._split_fit_data(*data)
        var_data = var_data.ravel()
        if len(var_data) > MAX_POINTS:
            var_data, func_data = var_data[:MAX_POINTS], func_data[:MAX_POINTS]
        var_diff = var_data[1:] - var_data[:-1]
        # Zero data
        c = np.mean(func_data)
        func_data -= c
        # Uniform spacing
        if np.allclose(var_diff, var_diff[0]):
            var_data = var_data[:-1:2]
            func_data = (func_data[:-1:2] + func_data[1::2]) / 2
        # Non-uniform spacing
        else:
            func_data_interp = scipy.interpolate.interp1d(var_data, func_data)
            var_data = np.linspace(
                var_data.min(), var_data.max(), num=(len(var_data)+1)//2
            )
            func_data = func_data_interp(var_data)
        # FFT for frequency estimation
        var_diff = var_data[1] - var_data[0]
        freqs = np.fft.fftfreq(len(func_data), d=var_diff)
        fft = abs(np.fft.fft(func_data))
        f = abs(freqs[np.argmax(fft[1:]) + 1])  # removing DC
        # Estimate amplitude
        a = np.std(func_data) * np.sqrt(2)
        # Estimate phase
        var_max = var_data[np.argmax(func_data)]
        var_min = var_data[np.argmin(func_data)]
        phi_max = 2*np.pi * ((f * var_max + 0.5) % 1)
        phi_min = 2*np.pi * ((f * var_min) % 1)
        phi = np.mean([phi_max, phi_min])
        # Set p0
        self.p0 = [a, f, phi, c]

    def find_popt(self, *data, **kwargs):
        psuccess = super().find_popt(*data, **kwargs)
        if psuccess:
            # Enforce phase within [0, 2π)
            for pname in ["phi"]:
                if pname in self.pfit:
                    pidx = self.pfit[pname]
                    self._popt[pidx] = self._popt[pidx] % (2 * np.pi)
        return psuccess


def cosine_2d(
    var, amplitude_x, amplitude_y, frequency_x, frequency_y, phase_x, phase_y,
    offset=0.0
):
    return (
        amplitude_x * np.cos(2 * np.pi * frequency_x * var[0] + phase_x)
        + amplitude_y * np.cos(2 * np.pi * frequency_y * var[1] + phase_y)
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
        var_data, func_data, _ = self._split_fit_data(*data)
        var_data = var_data.ravel()
        idx_min, idx_max = np.argmin(func_data), np.argmax(func_data)
        if idx_min == idx_max:
            self.p0 = [0, func_data[idx_min]]
        else:
            fmin, fmax = func_data[idx_min], func_data[idx_max]
            vmin, vmax = var_data[idx_min], var_data[idx_max]
            a = (fmax - fmin) / (vmax - vmin)
            c = fmax - a * vmax
            self.p0 = [a, c]

    def find_popt(self, *data, **kwargs):
        var_data, func_data, _ = self._split_fit_data(*data)
        var_data = var_data.ravel()
        if np.allclose(np.abs(self.__call__(var_data) - func_data), 0):
            popt_for_fit = self.p0_for_fit
            self.popt_for_fit = popt_for_fit
            self.pcov_for_fit = np.zeros(
                (len(popt_for_fit), len(popt_for_fit)), dtype=float
            )
            return True
        else:
            return super().find_popt(*data, **kwargs)


def power_law_1d(x, amplitude, power, center=0, offset=0):
    r"""
    Power law in one dimension.

    .. math::
        a (x - x_0)^p + c
    """
    dx = x - center
    xpow = np.zeros_like(x)
    np.power(dx, power, out=xpow, where=(dx != 0))
    return amplitude * xpow + offset


class FitPowerLaw1d(ModelBase):

    """
    Fit class for :py:func:`power_law_1d`.

    Parameters
    ----------
    a (amplitude)
    p (power)
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitPowerLaw1d")
    P_ALL = ["a", "p"]
    P_DEFAULT = [1, 1]

    @staticmethod
    def _func(var, *p):
        return power_law_1d(var, *p)

    def find_p0(self, *data):
        var_data, func_data, _ = self._split_fit_data(*data)
        var_data = var_data.ravel()
        mask = var_data > 0
        var_data, func_data = var_data[mask], func_data[mask]
        sign = 1 if np.mean(func_data) > 0 else -1
        func_data *= sign
        mask2 = func_data > 0
        var_log, func_log = np.log(var_data[mask2]), np.log(func_data[mask2])
        _fit = FitLinear1d()
        _fit.find_p0(var_log, func_log)
        if _fit.find_popt(var_log, func_log):
            a = np.exp(_fit.c)
            p = _fit.a
            self.p0 = [a * sign, p]
