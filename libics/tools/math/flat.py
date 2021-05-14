import numpy as np
import scipy

from libics.env import logging
from libics.tools.math.models import ModelBase


###############################################################################
# Oscillating Functions
###############################################################################


def cosine_1d(var, amplitude, frequency, phase, offset=0.0):
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
            # Enforce positive amplitude
            for pname, pcorrect in [["a", "phi"]]:
                if np.all([x in self.pfit for x in [pname, pcorrect]]):
                    nidx, cidx = self.pfit[pname], self.pfit[pcorrect]
                    if self._popt[nidx] < 0:
                        self._popt[nidx] = abs(self._popt[nidx])
                        self._popt[cidx] = self._popt[cidx] + np.pi
            # Enforce phase within [0, 2π)
            for pname in ["phi"]:
                if pname in self.pfit:
                    pidx = self.pfit[pname]
                    self._popt[pidx] = self._popt[pidx] % (2 * np.pi)
        return psuccess


def cosine_2d(var, amplitude, frequency_x, frequency_y, phase, offset=0.0):
    return amplitude * np.cos(
        2 * np.pi * (frequency_x * var[0] + frequency_y * var[1]) + phase
    ) + offset


class FitCosine2d(ModelBase):

    """
    Fit class for :py:func:`cosine_2d`.

    Parameters
    ----------
    a (amplitude)
    fx, fy (frequency without 2π)
    phi (additive phase)
    c (offset)

    Properties
    ----------
    f : Vectorial frequency (fx, fy)
    angle: Frequency angle arctan(fy/fx)
    """

    LOGGER = logging.get_logger("libics.tools.math.flat.FitCosine2d")
    P_ALL = ["a", "fx", "fy", "phi", "c"]
    P_DEFAULT = [1, 1, 0, 0, 0]

    @staticmethod
    def _func(var, *p):
        return cosine_2d(var, *p)

    @property
    def f(self):
        return np.array([self.fx, self.fy])

    @property
    def angle(self):
        if self.fx == 0:
            return np.pi / 2
        else:
            return np.arctan(self.fy / self.fx)

    def find_p0(self, *data, MAX_LINES=16, MAX_POINTS=1024):
        var_data, func_data, _ = self._split_fit_data(*data)
        # Perform 1D fits
        _step = var_data.shape[2] // MAX_LINES, var_data.shape[1] // MAX_LINES
        _step = [max(1, _x) for _x in _step]
        v_data = [var_data[0][:, ::_step[0]].T, var_data[1][::_step[1]]]
        f_data = [func_data[:, ::_step[0]].T, func_data[::_step[1]]]
        a, f, phi, c = [], [], [], []
        for dim in range(2):
            _a, _freq, _phi, _c = [], [], [], []
            for i, (_x, _f) in enumerate(zip(v_data[dim], f_data[dim])):
                _fit = FitCosine1d()
                _fit.find_p0(_x, _f, MAX_POINTS=MAX_POINTS)
                if _fit.find_popt(_x, _f):
                    _a.append(_fit.a)
                    _freq.append(_fit.f)
                    _phi.append(_fit.phi)
                    _c.append(_fit.c)
                else:
                    self.LOGGER.debug("find_p0: 1D fit did not converge")
            # Analyze p0
            a.append(np.max(_a))
            phi.append(np.mean(_phi))
            c.append(np.mean(_c))
            if np.std(_freq) > np.mean(_freq) / 3:  # if consistent fits
                f.append(0)
            else:
                f.append(np.mean(_freq))
        # Set angular direction
        _xmin, _xmax = var_data[0].min(), var_data[0].max()
        _ymin, _ymax = var_data[1].min(), var_data[1].max()
        _xc, _yc = np.mean([_xmin, _xmax]), np.mean([_ymin, _ymax])
        _slope = f[1] / f[0]
        _offset = _yc - _slope * _xc
        _xmin = max(_xmin, (_ymin-_offset)/_slope)
        _xmax = min(_xmax, (_ymax-_offset)/_slope)
        _xinterp = np.linspace(_xmin, _xmax, num=16)
        _yinterp_pos = _slope * _xinterp + _yc - _slope * _xc
        _yinterp_neg = -_slope * _xinterp + _yc + _slope * _xc
        _f_pos = scipy.interpolate.griddata(
            (var_data[0].ravel(), var_data[1].ravel()), func_data.ravel(),
            (_xinterp, _yinterp_pos)
        )
        _f_neg = scipy.interpolate.griddata(
            (var_data[0].ravel(), var_data[1].ravel()), func_data.ravel(),
            (_xinterp, _yinterp_neg)
        )
        # Set p0
        a = np.mean(a)
        if np.std(_f_pos) >= np.std(_f_neg):
            fx, fy = f[0], f[1]
        else:
            fx, fy = f[0], -f[1]
        phi = sum(phi)
        c = np.mean(func_data)
        self.p0 = [a, fx, fy, phi, c]

    def find_popt(self, *data, **kwargs):
        psuccess = super().find_popt(*data, **kwargs)
        if psuccess:
            # Enforce positive amplitude
            for pname, pcorrect in [["a", "phi"]]:
                if np.all([x in self.pfit for x in [pname, pcorrect]]):
                    nidx, cidx = self.pfit[pname], self.pfit[pcorrect]
                    if self._popt[nidx] < 0:
                        self._popt[nidx] = abs(self._popt[nidx])
                        self._popt[cidx] = self._popt[cidx] + np.pi
            # Enforce phase within [0, 2π)
            for pname in ["phi"]:
                if pname in self.pfit:
                    pidx = self.pfit[pname]
                    self._popt[pidx] = self._popt[pidx] % (2 * np.pi)
        return psuccess


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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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

    LOGGER = logging.get_logger("libics.tools.math.flat.FitPowerLaw1d")
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


class FitPowerLaw1dCenter(ModelBase):

    """
    Fit class for :py:func:`power_law_1d` with center as fit variable.

    Parameters
    ----------
    a (amplitude)
    p (power)
    x0 (center)
    """

    LOGGER = logging.get_logger("libics.tools.math.flat.FitPowerLaw1dCenter")
    P_ALL = ["a", "p", "x0"]
    P_DEFAULT = [1, 1, 0]

    @staticmethod
    def _func(var, *p):
        return power_law_1d(var, *p)

    def find_p0(self, *data):
        _fit = FitPowerLaw1d()
        _fit.find_p0(*data)
        self.p0 = [_fit.a, _fit.p, 0]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def error_function(x, amplitude, center, width, offset=0):
    r"""
    Error function in one dimension.

    .. math::
        a \mathrm{erf}((x - x_0) / w) + c
    """
    return amplitude * scipy.special.erf((x - center) / width) + offset


class FitErrorFunction(ModelBase):

    """
    Fit class for :py:func:`error_function`.

    Parameters
    ----------
    a (amplitude)
    x0 (center)
    w (width)
    c (offset)
    """

    LOGGER = logging.get_logger("libics.tools.math.flat.FitErrorFunction")
    P_ALL = ["a", "x0", "w", "c"]
    P_DEFAULT = [1, 0, 1, 0]

    @staticmethod
    def _func(var, *p):
        return error_function(var, *p)

    def find_p0(self, *data):
        var_data, func_data, _ = self._split_fit_data(*data)
        var_data = var_data.ravel()
        func_data = scipy.ndimage.uniform_filter(
            func_data, size=max(3, len(func_data) // 12)
        )
        # Extract parameters from statistics
        c = np.mean(func_data)
        func_data -= c
        x0 = var_data[np.argmin(np.abs(func_data))]
        var_data -= x0
        a = np.mean([np.max(func_data), -np.min(func_data)])
        func_data /= a
        w = 2 * np.mean([
            -var_data[np.argmin(np.abs(func_data + 0.5))],
            var_data[np.argmin(np.abs(func_data - 0.5))]
        ])
        self.p0 = [a, x0, w, c]

    def find_popt(self, *data, **kwargs):
        psuccess = super().find_popt(*data, **kwargs)
        if psuccess:
            # Enforce positive width
            if "a" in self.pfit and "w" in self.pfit:
                pidx = self.pfit["w"]
                if self._popt[pidx] < 0:
                    nidx = self.pfit["a"]
                    self._popt[pidx] *= -1
                    self._popt[nidx] *= -1
        return psuccess


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def linear_step_function(x, amplitude, center, width, offset=0):
    r"""
    Linearly interpolated step function in one dimension.

    .. math::
        c - a (for x < x_0 - w),
        \frac{a}{w} (x - x_0) + c (for x_0 - w < x < x_0 + w),
        c + a (for x > x_0 + w)
    """
    dx = (x - center)
    if width < 0:
        width = abs(width)
        amplitude *= -1
    return amplitude * np.piecewise(
        dx, [dx <= -width, dx >= width], [-1, 1, lambda dx: dx / width]
    ) + offset


class FitLinearStepFunction(ModelBase):

    """
    Fit class for :py:func:`linear_step_function`.

    Parameters
    ----------
    a (amplitude)
    x0 (center)
    w (width)
    c (offset)
    """

    LOGGER = logging.get_logger("libics.tools.math.flat.FitLinearStepFunction")
    P_ALL = ["a", "x0", "w", "c"]
    P_DEFAULT = [1, 0, 1, 0]

    @staticmethod
    def _func(var, *p):
        return linear_step_function(var, *p)

    def find_p0(self, *data):
        var_data, func_data, _ = self._split_fit_data(*data)
        var_data = var_data.ravel()
        func_data = scipy.ndimage.uniform_filter(
            func_data, size=max(3, len(func_data) // 12)
        )
        # Extract parameters from statistics
        c = np.mean(func_data)
        func_data -= c
        x0 = var_data[np.argmin(np.abs(func_data))]
        var_data -= x0
        a = np.mean([np.max(func_data), -np.min(func_data)])
        func_data /= a
        w = 2 * np.mean([
            -var_data[np.argmin(np.abs(func_data + 0.5))],
            var_data[np.argmin(np.abs(func_data - 0.5))]
        ])
        self.p0 = [a, x0, w, c]

    def find_popt(self, *data, **kwargs):
        psuccess = super().find_popt(*data, **kwargs)
        if psuccess:
            # Enforce positive width
            if "a" in self.pfit and "w" in self.pfit:
                pidx = self.pfit["w"]
                if self._popt[pidx] < 0:
                    nidx = self.pfit["a"]
                    self._popt[pidx] *= -1
                    self._popt[nidx] *= -1
        return psuccess
