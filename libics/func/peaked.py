import numpy as np
from scipy import special

from libics.func import fit


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

    def find_init_param(self, var_data, func_data):
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


FitGaussian2dTilt.__doc__ = (gaussian_2d_tilt.__doc__
                             + "\n\n\n" + fit.FitParamBase.__doc__)


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


FitAiryDisk2d.__doc__ = (airy_disk_2d.__doc__
                         + "\n\n\n" + fit.FitParamBase.__doc__)
