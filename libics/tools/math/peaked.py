import numpy as np
from scipy import ndimage, special, stats

from libics.tools.math import fit


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


class FitGaussian1d(fit.FitParamBase):

    def __init__(self, fit_offset=True, **kwargs):
        super().__init__(
            gaussian_1d, 4 if fit_offset else 3, **kwargs
        )

    def find_init_param(self, var_data, func_data):
        """
        Algorithm: filter -> centering (max/mean) -> windowing -> statistics.

        Notes
        -----
        Currently only works for positive data.
        """
        # Algorithm start
        x = var_data.astype(
            float if np.issubdtype(var_data.dtype, np.integer)
            else var_data.dtype
        )
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
        w = np.sqrt(np.sum((x[idx_slice] - x0)**2 * _pdf[idx_slice]))
        a = np.max(f)
        c = f_min
        # Algorithm end
        fit_offset = False if len(self.param) == 3 else True
        if fit_offset:
            self.param = [a, x0, w, c]
        else:
            self.param = [a, x0, w]

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


class FitGaussian2dTilt(fit.FitParamBase):

    def __init__(self, fit_offset=True, **kwargs):
        super().__init__(
            gaussian_2d_tilt, 7 if fit_offset else 6, **kwargs
        )

    def _find_init_param_linear(self, var_data, func_data):
        """
        Algorithm: linear min/max approximation.
        """
        fit_offset = False if len(self.param) == 6 else True
        offset = func_data.min()
        xmin, xmax = var_data[0].min(), var_data[0].max()
        ymin, ymax = var_data[1].min(), var_data[1].max()
        self.param[0] = func_data.max() - offset
        self.param[1], self.param[2] = (xmin + xmax) / 2, (ymin + ymax) / 2
        self.param[3], self.param[4] = (xmin - xmax) / 2, (ymin - ymax) / 2
        self.param[5] = 0.0
        if fit_offset:
            self.param[6] = offset

    def _find_init_param_fit1d(self, var_data, func_data):
        """
        Algorithm: 1D profile fits.
        """
        # Perform 1D profile fits
        xx, fx = fit.split_fit_data(np.sum(func_data, axis=1))
        xy, fy = fit.split_fit_data(np.sum(func_data, axis=0))
        fit_1d = FitGaussian1d(fit_offset=True)
        fit_1d.find_init_param(xx[0], fx)
        fit_1d.find_fit(xx[0], fx)
        px = np.copy(fit_1d.param)
        fit_1d.find_init_param(xy[0], fy)
        fit_1d.find_fit(xy[0], fy)
        py = np.copy(fit_1d.param)
        # Use 1D fit parameters for 2D fit initial parameters
        ax, x0, wx, _ = px
        ay, y0, wy, _ = py
        a = np.mean([ax / wy, ay / wx]) / np.sqrt(2 * np.pi)
        phi = 0
        c = 0
        fit_offset = False if len(self.param) == 6 else True
        if fit_offset:
            self.param = [a, x0, y0, wx, wy, phi, c]
        else:
            self.param = [a, x0, y0, wx, wy, phi]

    def find_init_param(self, var_data, func_data, algorithm="linear"):
        """
        Parameters
        ----------
        algorithm : `str`
            `"linear", "fit1d"`.
        """
        if algorithm == "linear":
            self._find_init_param_linear(var_data, func_data)
        elif algorithm == "fit1d":
            self._find_init_param_fit1d(var_data, func_data)
        else:
            raise ValueError("invalid algorithm ({:s})".format(algorithm))

    def find_fit(self, *args, **kwargs):
        ret = super().find_fit(*args, **kwargs)
        self.param[3] = np.abs(self.param[3])
        self.param[4] = np.abs(self.param[4])
        return ret


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


class FitLorentzian1dAbs(fit.FitParamBase):

    def __init__(self, fit_offset=True, **kwargs):
        super().__init__(
            lorentzian_1d_abs, 4 if fit_offset else 3, **kwargs
        )

    def find_init_param(self, var_data, func_data):
        """
        Algorithm: dummy max.
        """
        idx_max = np.argmax(func_data)
        self.param[0] = func_data[idx_max]
        self.param[1] = var_data[idx_max]
        self.param[2] = (np.max(var_data) - np.min(var_data)) / 4
        if len(self.param) == 4:
            self.param[3] = 0.0


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


class FitAiryDisk2d(fit.FitParamBase):

    def __init__(self, fit_offset=True, **kwargs):
        super().__init__(
            airy_disk_2d, 5 if fit_offset else 4, **kwargs
        )

    def find_init_param(self, var_data, func_data):
        """
        Algorithm: linear min/max approximation.
        """
        fit_offset = False if len(self.param) == 4 else True
        offset = func_data.min()
        xmin, xmax = var_data[0].min(), var_data[0].max()
        ymin, ymax = var_data[1].min(), var_data[1].max()
        self.param[0] = func_data.max() - offset
        self.param[1], self.param[2] = (xmin + xmax) / 2, (ymin + ymax) / 2
        self.param[3] = (xmax - xmin + ymax - ymin) / 10
        self.param[4] = 0.0
        if fit_offset:
            self.param[4] = offset


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
