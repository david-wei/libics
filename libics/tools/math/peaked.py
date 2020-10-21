import numpy as np
from scipy import ndimage, special, stats

from libics.env import logging
from libics.tools.math.models import ModelBase


###############################################################################
# Exponential Functions
###############################################################################


def exponential_decay_1d(x,
                         amplitude, center, length, offset=0.0):
    r"""
    Exponential decay in one dimension.

    .. math::
        A e^{-\frac{|x - c|}{\xi}} + C

    Parameters
    ----------
    x : `float`
        Variable :math:`x`.
    amplitude : `float`
        Amplitude :math:`A`.
    center : `float`
        Center :math:`c`.
    length : `float`
        Characteristic length :math:`\xi`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`
    """
    exponent = -np.abs(x - center) / length
    return amplitude * np.exp(exponent) + offset


class FitExponentialDecay1d(ModelBase):

    """
    Fit class for :py:func:`exponential_decay_1d`.

    Parameters
    ----------
    a (amplitude)
    x0 (center)
    xi (length)
    c (offset)
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitExponentialDecay1d")
    P_ALL = ["a", "x0", "xi", "c"]
    P_DEFAULT = [1, 0, 1, 0]

    @staticmethod
    def _func(var, *p):
        return exponential_decay_1d(var, *p)

    def find_p0(self, *data):
        var_data, func_data = self._split_fit_data(*data)
        c = np.min(func_data) / 2
        a = np.max(func_data) - c
        x0 = 0
        xi = (np.max(var_data) - np.min(var_data)) / np.log(np.abs(2 * c / a))
        self.p0 = [a, x0, xi, c]


def exponential_decay_nd(x,
                         amplitude, center, length, offset=0.0):
    r"""
    Exponential decay in :math:`n` dimensions.

    .. math::
        A e^{-\sum_{i=1}^n \frac{|x_i - c_i|}{\xi_i}} + C

    Parameters
    ----------
    x : `numpy.array(n, float)`
        Variables :math:`x_i`.
    amplitude : `float`
        Amplitude :math:`A`.
    center : `numpy.array(n, float)`
        Centers :math:`c_i`.
    length : `numpy.array(n, float)`
        Characteristic lengths :math:`\xi_i`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`
    """
    exponent = -np.sum(np.abs(x - center) / length)
    return amplitude * np.exp(exponent) + offset


###############################################################################
# Gaussian Functions
###############################################################################


def gaussian_1d(x,
                amplitude, center, width, offset=0.0):
    r"""
    Gaussian in one dimension.

    .. math::
        A e^{-\frac{(x - c)^2}{2 \sigma^2}} + C

    Parameters
    ----------
    x : `float`
        Variable :math:`x`.
    amplitude : `float`
        Amplitude :math:`A`.
    center : `float`
        Center :math:`c`.
    width : `float`
        Width :math:`\sigma`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    exponent = -((x - center) / width)**2 / 2.0
    return amplitude * np.exp(exponent) + offset


class FitGaussian1d(ModelBase):

    """
    Fit class for :py:func:`gaussian_1d`.

    Parameters
    ----------
    a (amplitude)
    x0 (center)
    wx (width)
    c (offset)
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitGaussian1d")
    P_ALL = ["a", "x0", "wx", "c"]
    P_DEFAULT = [1, 0, 1, 0]

    @staticmethod
    def _func(var, *p):
        return gaussian_1d(var, *p)

    def find_p0(self, *data):
        """
        Algorithm: filter -> centering (max/mean) -> windowing -> statistics.

        Notes
        -----
        Currently only works for positive data.
        """
        var_data, func_data = self._split_fit_data(*data)
        # Algorithm start
        x = var_data.astype(
            float if np.issubdtype(var_data.dtype, np.integer)
            else var_data.dtype
        ).ravel()
        f = func_data.astype(
            float if np.issubdtype(func_data.dtype, np.integer)
            else func_data.dtype
        )
        f_min = np.min(f)
        # 1D density as probability distribution
        _pdf = ndimage.uniform_filter1d(f - f_min, 2)
        _pdf[_pdf < 0] = 0
        _pdf /= np.sum(_pdf)
        # Initial parameters
        x0 = self._get_center(x, f)
        # Avoid standard deviation bias
        idx0 = np.argmin(np.abs(x - x0))
        idx_slice = None
        if len(x) - idx0 > idx0:
            idx_slice = slice(None, 2 * idx0 + 1)
        else:
            idx_slice = slice(2 * idx0 - len(x), None)
        wx = np.sqrt(np.sum((x[idx_slice] - x0)**2 * _pdf[idx_slice]))
        a = np.max(f)
        c = f_min
        # Algorithm end
        self.p0 = [a, x0, wx, c]

    @staticmethod
    def _get_center(x, f, max_weight=0.3, use_unbiased_pdf=True):
        """
        Finds the center using max and mean.

        Assumes that data is positive.

        Parameters
        ----------
        x, f : `np.ndarray(1)`
            Variable and function data.
        max_weight : `float`
            Center is calculated as weighted average between
            data maximum and mean. `max_weight` selects the
            relative weight of the maximum.
        use_unbiased_pdf : `bool`
            Flag whether to use symmetric statistics around maximum.
        """
        # Direct maximum
        idx_max = np.argmax(f)
        x_max = x[idx_max]
        # Statistical mean
        pdf = np.copy(f)
        if use_unbiased_pdf:
            idx_slice = None
            if len(x) - idx_max > idx_max:
                idx_slice = slice(None, 2 * idx_max + 1)
            else:
                idx_slice = slice(2 * idx_max - len(x), None)
            pdf = pdf[idx_slice]
            x = np.copy(x)[idx_slice]
        pdf[pdf < 0] = 0
        pdf = pdf / np.sum(pdf)
        x_stat = np.sum(x * pdf)
        # Average
        x0 = (1 - max_weight) * x_stat + max_weight * x_max
        return x0


def gaussian_nd(x,
                amplitude, center, width, offset=0.0):
    r"""
    Gaussian in :math:`n` dimensions.

    .. math::
        A e^{-\sum_{i=1}^n \left( \frac{x_i - c_i}{2 \sigma_i} \right)^2} + C

    Parameters
    ----------
    x : `numpy.array(n, float)`
        Variables :math:`x_i`.
    amplitude : `float`
        Amplitude :math:`A`.
    center : `numpy.array(n, float)`
        Centers :math:`c_i`.
    width : `numpy.array(n, float)`
        Widths :math:`\sigma_i`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    exponent = -np.sum(((x - center) / width)**2) / 2.0
    return amplitude * np.exp(exponent) + offset


def gaussian_nd_symmetric(x,
                          amplitude, center, width, offset=0.0):
    r"""
    Symmetric Gaussian in :math:`n` dimensions.

    .. math::
        A e^{-\sum_{i=1}^n \left( \frac{x_i - c_i}{2 \sigma} \right)^2} + C

    Parameters
    ----------
    x : `numpy.array(n, float)`
        Variables :math:`x_i`.
    amplitude : `float`
        Amplitude :math:`A`.
    center : `numpy.array(n, float)`
        Centers :math:`c_i`.
    width : `numpy.array(n, float)`
        Width :math:`\sigma`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    exponent = -np.sum((x - center)**2) / 2.0 / width**2
    return amplitude * np.exp(exponent) + offset


def gaussian_nd_centered(x,
                         amplitude, width, offset=0.0):
    r"""
    Centered Gaussian in :math:`n` dimensions.

    .. math::
        A e^{-\sum_{i=1}^n \left( \frac{x_i}{2 \sigma_i} \right)^2} + C

    Parameters
    ----------
    x : `numpy.array(n, float)`
        Variables :math:`x_i`.
    amplitude : `float`
        Amplitude :math:`A`.
    width : `numpy.array(n, float)`
        Widths :math:`\sigma_i`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    exponent = -np.sum((x / width)**2) / 2.0
    return amplitude * np.exp(exponent) + offset


def gaussian_2d_tilt(
    var, amplitude, center_x, center_y, width_u, width_v, tilt, offset=0.0
):
    r"""
    Tilted (rotated) Gaussian in two dimensions.

    .. math::
        A e^{-\frac{((x - x_0) \cos \theta + (y - y_0) \sin \theta))^2}
                   {2 \sigma_u^2}
             -\frac{((y - y_0) \cos \theta - (x - x_0) \sin \theta))^2}
                   {2 \sigma_v^2}} + C

    Parameters
    ----------
    var : `numpy.array(2, float)`
        Variables :math:`x, y`.
    amplitude : `float`
        Amplitude :math:`A`.
    center_x, center_y : `float`
        Centers :math:`x_0, y_0`.
    width_u, width_v : `float`
        Widths :math:`\sigma_u, \sigma_v`.
    tilt : `float(0, numpy.pi)`
        Tilt angle :math:`\theta`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    tilt_cos, tilt_sin = np.cos(tilt), np.sin(tilt)
    dx, dy = var[0] - center_x, var[1] - center_y
    exponent = (
        ((dx * tilt_cos + dy * tilt_sin) / width_u)**2
        + ((dy * tilt_cos - dx * tilt_sin) / width_v)**2
    )
    return amplitude * np.exp(-exponent / 2.0) + offset


class FitGaussian2dTilt(ModelBase):

    """
    Fit class for :py:func:`gaussian_2d_tilt`.

    Parameters
    ----------
    a (amplitude)
    x0, y0 (center)
    wu, wv (width)
    tilt (tilt in [-45°, 45°])
    c (offset)
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitGaussian2dTilt")
    P_ALL = ["a", "x0", "y0", "wu", "wv", "tilt", "c"]
    P_DEFAULT = [1, 0, 0, 1, 1, 0, 0]

    @staticmethod
    def _func(var, *p):
        return gaussian_2d_tilt(var, *p)

    def _find_p0_linear(self, var_data, func_data):
        """
        Algorithm: linear min/max approximation.
        """
        c = func_data.min()
        xmin, xmax = var_data[0].min(), var_data[0].max()
        ymin, ymax = var_data[1].min(), var_data[1].max()
        a = func_data.max() - c
        x0, y0 = (xmin + xmax) / 2, (ymin + ymax) / 2
        wu, wv = (xmin - xmax) / 2, (ymin - ymax) / 2
        tilt = 0
        return [a, x0, y0, wu, wv, tilt, c]

    def _find_p0_fit1d(self, var_data, func_data):
        """
        Algorithm: 1D profile fits.
        """
        # Perform 1D profile fits
        len_x, len_y = func_data.shape
        xdata = np.sum(func_data, axis=1)
        ydata = np.sum(func_data, axis=0)
        fit_1d = FitGaussian1d()
        fit_1d.find_p0(xdata)
        fit_1d.find_popt(xdata)
        px = np.copy(fit_1d.popt)
        fit_1d.find_p0(ydata)
        fit_1d.find_popt(ydata)
        py = np.copy(fit_1d.popt)
        # Use 1D fit parameters for 2D fit initial parameters
        ax, x0, wx, cx = px
        ay, y0, wy, cy = py
        a = np.mean([ax / wy, ay / wx]) / np.sqrt(2 * np.pi)
        tilt = 0
        c = np.mean([cx / len_y, cy / len_x])
        return [a, x0, y0, wx, wy, tilt, c]

    def find_p0(self, *data, algorithm="fit1d"):
        """
        Parameters
        ----------
        algorithm : `str`
            `"linear", "fit1d"`.
        """
        var_data, func_data = self._split_fit_data(*data)
        if algorithm == "linear":
            self.p0 = self._find_p0_linear(var_data, func_data)
        elif algorithm == "fit1d":
            self.p0 = self._find_p0_fit1d(var_data, func_data)
        else:
            raise ValueError("invalid algorithm ({:s})".format(algorithm))

    def find_popt(self, *args, **kwargs):
        psuccess = super().find_popt(*args, **kwargs)
        if psuccess:
            # Enforce positive widths
            for pname in ["wu", "wv"]:
                if pname in self.pfit:
                    pidx = self.pfit[pname]
                    self._popt[pidx] = np.abs(self._popt[pidx])
            # Enforce tilt angle in [-45°, 45°]
            if "tilt" in self.pfit:
                tilt_idx = self.pfit["tilt"]
                tilt = self.popt_for_fit[tilt_idx] % np.pi
                # Tilt angle in [135°, 180°]
                if tilt >= 3/4 * np.pi:
                    tilt -= np.pi
                # Tilt angle in [45°, 135°]
                elif tilt > 1/4 * np.pi:
                    # Perform wu/wv axes swap
                    if np.all((pname in self.pfit for pname in ["wu", "wv"])):
                        tilt -= 1/2 * np.pi
                        wu_idx, wv_idx = self.pfit["wu"], self.pfit["wv"]
                        popt = self.popt_for_fit
                        pcov = self.pcov_for_fit
                        popt[[wu_idx, wv_idx]] = popt[[wv_idx, wu_idx]]
                        pcov[[wu_idx, wv_idx]] = pcov[[wu_idx, wv_idx]]
                        pcov[:, [wu_idx, wv_idx]] = pcov[:, [wu_idx, wv_idx]]
                        self.popt_for_fit = popt
                        self.pcov_for_fit = pcov
                self.popt_for_fit[tilt_idx] = tilt
        return psuccess


###############################################################################
# Lorentzian Functions
###############################################################################


def lorentzian_1d_complex(
    var, amplitude, center, width, offset=0.0
):
    return amplitude / (1j + (var - center) / width) + offset


def lorentzian_1d_abs(
    var, amplitude, center, width, offset=0.0
):
    return amplitude / (1 + ((var - center) / width)**2) + offset


class FitLorentzian1dAbs(ModelBase):

    """
    Fit class for :py:func:`lorentzian_1d_abs`.

    Parameters
    ----------
    a (amplitude)
    x0 (center)
    wx (width)
    c (offset)
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitLorentzian1dAbs")
    P_ALL = ["a", "x0", "wx", "c"]
    P_DEFAULT = [1, 0, 1, 0]

    @staticmethod
    def _func(var, *p):
        return lorentzian_1d_abs(var, *p)

    def find_p0(self, *data):
        """
        Algorithm: dummy max.
        """
        var_data, func_data = self._split_fit_data(*data)
        idx_max = np.argmax(func_data)
        a = func_data[idx_max]
        x0 = var_data[idx_max]
        wx = (np.max(var_data) - np.min(var_data)) / 4
        c = 0
        self.p0 = [a, x0, wx, c]


###############################################################################
# Oscillating Functions
###############################################################################


def airy_disk_2d(
    var, amplitude, center_x, center_y, width, offset=0.0
):
    r"""
    .. math::
        A \left( \frac{2 J_1 \left( \sqrt{(x-x_0)^2 + (y-y_0)^2} / w \right)}
                      {\sqrt{(x-x_0)^2 + (y-y_0)^2} / w} \right)^2 + C

    Parameters
    ----------
    `var` :math:`(x, y)`, `amplitude` :math:`A`,
    `center_(x, y)` :math:`(x_0, y_0), `width` :math:`w`,
    `offset` :math:`C`.

    Notes
    -----
    Handles limiting value at :math:`(x, y) -> (0, 0)`.
    """
    arg = np.sqrt((var[0] - center_x)**2 + (var[1] - center_y)**2) / width
    res = np.piecewise(
        arg, [arg < 1e-8, arg >= 1e-8],
        [lambda x: 1, lambda x: 2 * special.j1(x) / x]
    )
    return amplitude * res**2 + offset


class FitAiryDisk2d(ModelBase):

    """
    Fit class for :py:func:`airy_disk_2d`.

    Parameters
    ----------
    a (amplitude)
    x0, y0 (center)
    w (width)
    c (offset)
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitAiryDisk2d")
    P_ALL = ["a", "x0", "y0", "w", "c"]
    P_DEFAULT = [1, 0, 0, 1, 0]

    @staticmethod
    def _func(var, *p):
        return airy_disk_2d(var, *p)

    def find_p0(self, *data):
        """
        Algorithm: linear min/max approximation.
        """
        var_data, func_data = self._split_fit_data(*data)
        c = func_data.min()
        xmin, xmax = var_data[0].min(), var_data[0].max()
        ymin, ymax = var_data[1].min(), var_data[1].max()
        a = func_data.max() - c
        x0, y0 = (xmin + xmax) / 2, (ymin + ymax) / 2
        w = (xmax - xmin + ymax - ymin) / 10
        self.p0 = [a, x0, y0, w, c]


def dsc_bloch_osc_1d(
    var, hopping, gradient
):
    r"""
    .. math::
        J_n^2 \left( \frac{4 J}{\Delta}
        \cdot \left| \sin \frac{\Delta t}{2} \right| \right)

    :math:`\Delta = F a` specifies the energy difference between neighbouring
    sites with distance :math:`a` at a potential gradient :math:`F`.
    Energies :math:`J, \Delta` should be given in :math:`2 \pi` frequency
    units, i.e. in units of :math:`\hbar`.

    Parameters
    ----------
    `var` :math:`(n, t)`, `hopping` :math:`J`, `gradient` :math:`\Delta`.

    Returns
    -------
    res
        Probability distribution of 1D Bloch oscillations for the given
        site `var[0]` at the given time `var[1]`.
    """
    arg = np.abs(np.sin(gradient * var[-1] / 2))
    return special.jv(var[0], 4 * hopping / gradient * arg)**2


class RndDscBlochOsc1d(stats.rv_discrete):
    """
    1D Bloch oscillation site occupation random variable.

    Parameters
    ----------
    site : `tuple(int)`
        Minimum and maximum sites `(site_min, site_max)`.
    time : `float`
        Evolution time in seconds (s).
    hopping : `float`
        Hopping frequency in radians (rad).
    gradient : `float`
        Frequency difference between neighbouring sites in radians (rad).
    """

    def __new__(
        cls, *args,
        sites=None, time=None, hopping=None, gradient=None, **kwargs
    ):
        return super().__new__(cls, *args, name="bloch_osc_1d", **kwargs)

    def __init__(
        self, *args,
        sites=(-100, 100), time=0, hopping=2*np.pi, gradient=2*np.pi, **kwargs
    ):
        super().__init__(*args, name="bloch_osc_1d", **kwargs)
        self.sites = sites
        self.time = time
        self.hopping = hopping
        self.gradient = gradient

    def _get_support(self, *args):
        return self.sites

    def _pmf(self, x):
        t = np.full_like(x, self.time, dtype=float)
        return dsc_bloch_osc_1d([x, t], self.hopping, self.gradient)


def dsc_bloch_osc_2d(
    var, hopping_x, hopping_y, gradient_x, gradient_y
):
    """
    See :py:func:`bloch_osc_1d`.

    `var` :math:`(n_x, n_y, t)`
    """
    arg_x = np.abs(np.sin(gradient_x * var[-1] / 2))
    arg_y = np.abs(np.sin(gradient_y * var[-1] / 2))
    prb_x = special.jv(var[0], 4 * hopping_x / gradient_x * arg_x)**2
    prb_y = special.jv(var[1], 4 * hopping_y / gradient_y * arg_y)**2
    return prb_x * prb_y


def dsc_bloch_osc_3d(
    var, hopping_x, hopping_y, hopping_z, gradient_x, gradient_y, gradient_z
):
    """
    See :py:func:`bloch_osc_1d`.

    `var` :math:`(n_x, n_y, n_z, t)`
    """
    arg_x = np.abs(np.sin(gradient_x * var[-1] / 2))
    arg_y = np.abs(np.sin(gradient_y * var[-1] / 2))
    arg_z = np.abs(np.sin(gradient_z * var[-1] / 2))
    prb_x = special.jv(var[0], 4 * hopping_x / gradient_x * arg_x)**2
    prb_y = special.jv(var[1], 4 * hopping_y / gradient_y * arg_y)**2
    prb_z = special.jv(var[2], 4 * hopping_z / gradient_z * arg_z)**2
    return prb_x * prb_y * prb_z


def dsc_ballistic_1d(var, hopping):
    r"""
    .. math::
        J_n^2 \left( 2 J t \right)

    The hopping energy :math:`J` should be given in :math:`2 \pi` frequency
    units, i.e. in units of :math:`\hbar`.

    Parameters
    ----------
    `var` :math:`(n, t)`, `hopping` :math:`J`.

    Returns
    -------
    res
        Probability distribution of 1D ballistic transport for the given
        site `var[0]` at the given time `var[1]`.
    """
    return special.jv(var[0], 2 * hopping * var[-1])**2


class RndDscBallistic1d(stats.rv_discrete):
    """
    1D ballistic transport site occupation random variable.

    Parameters
    ----------
    site : `tuple(int)`
        Minimum and maximum sites `(site_min, site_max)`.
    time : `float`
        Evolution time in seconds (s).
    hopping : `float`
        Hopping frequency in radians (rad).
    """

    def __new__(
        cls, *args,
        sites=None, time=None, hopping=None, **kwargs
    ):
        return super().__new__(cls, *args, name="ballistic_1d", **kwargs)

    def __init__(
        self, *args,
        sites=(-100, 100), time=0, hopping=2*np.pi, **kwargs
    ):
        super().__init__(*args, name="ballistic_1d", **kwargs)
        self.sites = sites
        self.time = time
        self.hopping = hopping

    def _get_support(self, *args):
        return self.sites

    def _pmf(self, x):
        t = np.full_like(x, self.time, dtype=float)
        return dsc_ballistic_1d([x, t], self.hopping)


def dsc_diffusive_1d(var, diffusion):
    r"""
    .. math::
        \frac{1}{\sqrt{4 \pi D t}}
        \exp \left( -\frac{n^2}{4 D t} \right)

    The diffusion constant :math:`D` should be given in units of the
    lattice constant :math:`a`, i.e. :math:`D_\text{SI} = D / a^2`.

    Parameters
    ----------
    `var` :math:`(n, t)`, `hopping` :math:`J`.

    Returns
    -------
    res
        Probability distribution of 1D ballistic transport for the given
        site `var[0]` at the given time `var[1]`.
    """
    _is_nonzero = (var[-1] != 0)  # Handling time division by zero
    result = None
    if np.isscalar(_is_nonzero):
        if _is_nonzero:
            _width = np.sqrt(2 * diffusion * var[-1])
            _amplitude = 1 / np.sqrt(4 * np.pi * diffusion * var[-1])
            result = gaussian_1d(var[0], _amplitude, 0, _width, offset=0)
        else:
            result = 1 if var[0] == 0 else 0
    else:
        _width = np.sqrt(2 * diffusion * var[-1][_is_nonzero])
        _amplitude = 1 / np.sqrt(4 * np.pi * diffusion * var[-1][_is_nonzero])
        result = np.zeros_like(var[0], dtype=float)
        result[(~_is_nonzero) & (var[0] == 0)] = 1
        result[_is_nonzero] = gaussian_1d(
            var[0][_is_nonzero], _amplitude, 0, _width, offset=0
        )
    return result


class RndDscDiffusive1d(stats.rv_discrete):
    """
    1D random variable for diffusive transport on discrete sites.

    Parameters
    ----------
    site : `tuple(int)`
        Minimum and maximum sites `(site_min, site_max)`.
    time : `float`
        Evolution time in seconds (s).
    diffusion : `float`
        Diffusion constant in sites² per second (1/s).
    """

    def __new__(
        cls, *args,
        sites=None, time=None, diffusion=None, **kwargs
    ):
        return super().__new__(cls, *args, name="diffusive_1d", **kwargs)

    def __init__(
        self, *args,
        sites=(-100, 100), time=0, diffusion=1, **kwargs
    ):
        super().__init__(*args, name="diffusive_1d", **kwargs)
        self.sites = sites
        self.time = time
        self.diffusion = diffusion

    def _get_support(self, *args):
        return self.sites

    def _pmf(self, x):
        t = np.full_like(x, self.time, dtype=float)
        return dsc_diffusive_1d([x, t], self.diffusion)


###############################################################################
# Miscellaneous distribution functions
###############################################################################


def gamma_distribution_1d(x,
                          amplitude, mean, number, offset=0.0):
    r"""
    Gamma distribution in one dimension.

    .. math::
        A \frac{(N / \mu)^N}{\Gamma (N)} x^{N - 1}
        e^{-\frac{N}{\mu} x} + C

    Parameters
    ----------
    x : `float`
        Variable :math:`x`.
    amplitude : `float`
        Amplitude :math:`A`.
    mean : `float`
        Mean :math:`\mu`.
    number : `float`
        Number :math:`N`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    exponent = (
        number * (np.log(number) - np.log(mean) - x / mean)
        + (number - 1) * np.log(x)
        - special.gammaln(number)
    )
    return amplitude * np.exp(exponent) + offset
