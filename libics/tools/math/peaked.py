import numpy as np
from scipy import ndimage, optimize, special, signal, stats

from libics.env import logging
from libics.tools.math.models import ModelBase, RvContinuous


###############################################################################
# Exponential Functions
###############################################################################


def exponential_1d(x, amplitude, rate, offset=0.0):
    r"""
    Exponential function in one dimension.

    .. math::
        A e^{\gamma x} + C

    Parameters
    ----------
    x : `float`
        Variable :math:`x`.
    amplitude : `float`
        Amplitude :math:`A`.
    rate : `float`
        Rate of exponential :math:`\gamma`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`
    """
    return amplitude * np.exp(rate * x) + offset


class FitExponential1d(ModelBase):

    """
    Fit class for :py:func:`exponential_1d`.

    Parameters
    ----------
    a : `float`
        amplitude
    g : `float`
        rate
    c : `float`
        offset

    Attributes
    ----------
    x0 : `float`
        variable offset, assuming unity amplitude
    xi : `float`
        decay length, inverse rate
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitExponential1d")
    P_ALL = ["a", "g", "c"]
    P_DEFAULT = [1, 1, 0]

    @staticmethod
    def _func(var, *p):
        return exponential_1d(var, *p)

    @property
    def x0(self):
        return -np.log(np.abs(self.a)) / self.g

    @property
    def xi(self):
        return 1 / self.g

    def find_p0(self, *data):
        var_data, func_data, _ = self.split_fit_data(*data)
        var_data = var_data.ravel()
        # Smoothened derivatives
        func_data_filter = ndimage.uniform_filter(
            func_data, size=max(3, len(func_data) // 12)
        )
        first_derivative = np.gradient(func_data_filter, var_data)
        second_derivative = np.gradient(
            ndimage.uniform_filter(first_derivative, size=3), var_data
        )
        mask = first_derivative != 0
        # Extract parameters
        g = np.median(second_derivative[mask] / first_derivative[mask])
        _exp_gx = np.exp(g * var_data)
        a = np.median(first_derivative / g / _exp_gx)
        c = np.median(func_data_filter - a * _exp_gx)
        self.p0 = [a, g, c]


def exponential_decay_1d(*args, **kwargs):
    raise RuntimeError("DEPRECATED: use function `exponential_1d`")


class FitExponentialDecay1d:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("DEPRECATED: use class `FitExponential1d`")


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


class _Normal1dDistribution_gen(
    RvContinuous, stats._continuous_distns.norm_gen
):

    def __init__(self, *args, **kwargs):
        RvContinuous.__init__(self, *args, **kwargs)
        stats._continuous_distns.norm_gen.__init__(self, *args, **kwargs)

    def _ipdf(self, p, branch="left"):
        sign = -1 if branch == "left" else 1
        return np.sqrt(2 * np.log(p * np.sqrt(2 * np.pi))) * sign

    def _mode(self):
        return 0.0

    def _amplitude(self):
        return 1 / np.sqrt(2 * np.pi)


Normal1dDistribution = _Normal1dDistribution_gen()


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
    if np.isscalar(exponent):
        val = 0
        if exponent > -700:
            val = np.exp(exponent)
    else:
        val = np.zeros_like(exponent)
        np.exp(exponent, out=val, where=(exponent > -700))
    return amplitude * val + offset


class FitGaussian1d(ModelBase):

    """
    Fit class for :py:func:`gaussian_1d`.

    Parameters
    ----------
    a : `float`
        amplitude
    x0 : `float`
        center
    wx : `float`
        width
    c : `float`
        offset
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitGaussian1d")
    P_ALL = ["a", "x0", "wx", "c"]
    P_DEFAULT = [1, 0, 1, 0]
    DISTRIBUTION = Normal1dDistribution

    @staticmethod
    def _func(var, *p):
        return gaussian_1d(var, *p)

    @staticmethod
    def _find_p0_stat(var_data, func_data):
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
        x0 = FitGaussian1d._get_center(x, _pdf)
        # Avoid standard deviation bias
        idx0 = np.argmin(np.abs(x - x0))
        idx_slice = None
        # If insufficient statistics
        if min(idx0, len(x) - idx0) < 5:
            pass
        elif len(x) - idx0 > idx0:
            idx_slice = slice(None, 2 * idx0 + 1)
        else:
            idx_slice = slice(2 * idx0 - len(x), None)
        wx = np.sqrt(np.sum((x[idx_slice] - x0)**2 * _pdf[idx_slice]))
        c = f_min
        a = np.max(f) - c
        # Algorithm end
        p0 = [a, x0, wx, c]
        return p0

    def find_p0(self, *data):
        var_data, func_data, _ = self.split_fit_data(*data)
        self.p0 = self._find_p0_stat(var_data, func_data)

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

    def find_popt(self, *args, maxfev=100000, **kwargs):
        if maxfev is not None:
            kwargs["maxfev"] = maxfev
        psuccess = super().find_popt(*args, **kwargs)
        if psuccess:
            # Enforce positive width
            for pname in ["wx"]:
                if pname in self.pfit:
                    pidx = self.pfit[pname]
                    self._popt[pidx] = np.abs(self._popt[pidx])
            psuccess &= np.all([
                getattr(self, f"{pname}_std") / getattr(self, pname) < 1
                for pname in ["a", "wx"]
            ])
        return psuccess

    def get_distribution(self):
        return self.DISTRIBUTION(loc=self.x0, scale=self.wx)

    @property
    def distribution_amplitude(self):
        return self.a / self.DISTRIBUTION.amplitude(loc=self.x0, scale=self.wx)


class _SkewNormal1dDistribution_gen(RvContinuous):

    """
    Distribution class for the skew normal distribution.

    Uses the parameterization of:
    https://en.wikipedia.org/wiki/Skew_normal_distribution.
    """

    LOGGER = logging.get_logger(
        "libics.tools.math.peaked.SkewNormal1dDistribution"
    )

    def _argcheck(self, *args):
        return len(args) == 1

    def _pdf(self, x, alpha):
        xi = x / np.sqrt(2)
        if np.isscalar(xi):
            val = 0
            if np.abs(xi) < 26:
                val = np.exp(-xi**2)
        else:
            val = np.zeros_like(xi)
            np.exp(-xi**2, out=val, where=(np.abs(xi) < 26))
        return (
            1 / np.sqrt(2 * np.pi) * val
            * (1 + special.erf(alpha * xi))
        )

    def _cdf(self, x, alpha):
        xi = x / np.sqrt(2)
        phi = (1 + special.erf(xi)) / 2
        t = special.owens_t(xi * np.sqrt(2), alpha)
        return phi - 2 * t

    def _ppf(self, q, alpha):
        if np.isscalar(q):
            if q <= 0:
                return -np.inf
            elif q >= 1:
                return np.inf
            x1 = self._mean(alpha)
            if x1 == 0:
                x1 = np.sign(alpha + 1e-50)
            res = optimize.root_scalar(
                lambda x: self._cdf(x, alpha) - q, x0=0, x1=x1
            ).root
        else:
            res = np.zeros_like(q, dtype=float)
            mask0, mask1 = (q <= 0), (q >= 1)
            res[mask0] = -np.inf
            res[mask1] = np.inf
            mask = (~mask0) & (~mask1)
            q_opt = q[mask]
            if len(q_opt) > 0:
                x0_opt = np.zeros_like(q_opt, dtype=float)
                res[mask] = optimize.root(
                    lambda x: self._cdf(x, alpha) - q_opt, x0_opt
                ).x
        return res

    def _stats(self, alpha):
        delta = alpha / np.sqrt(1 + alpha**2)
        mu = delta * np.sqrt(2 / np.pi)
        sigma = np.sqrt(1 - mu**2)
        mean = delta * np.sqrt(2 / np.pi)
        variance = 1 - mu**2
        skewness = (4 - np.pi) / 2 * (mu / sigma)**3
        kurtosis_exc = 2 * (np.pi - 3) * (mu / sigma)**4
        return mean, variance, skewness, kurtosis_exc

    def _mode(self, alpha):
        delta = alpha / np.sqrt(1 + alpha**2)
        mu = np.sqrt(2 / np.pi) * delta
        sigma = np.sqrt(1 - mu**2)
        if np.isscalar(alpha):
            alpha_abs_inv = (
                np.inf if np.abs(alpha) < 1e-2 else 1 / np.abs(alpha)
            )
        else:
            alpha_abs_inv = np.full_like(alpha, np.inf)
            np.divide(1, np.abs(alpha), out=alpha_abs_inv,
                      where=(np.abs(alpha) < 1e-2))
        return (
            mu - self.skewness(alpha) * sigma / 2
            - np.sign(alpha) / 2 * np.exp(-2 * np.pi * alpha_abs_inv)
        )

    def cv_skewness_to_alpha(self, skewness):
        k = skewness * 2 / (4 - np.pi)
        k3 = np.sign(k) * np.abs(k)**(1/3)
        mu = k3 / np.sqrt(1 + k3**2)
        delta = mu * np.sqrt(np.pi / 2)
        alpha = delta / np.sqrt(1 - delta**2)
        return alpha


SkewNormal1dDistribution = _SkewNormal1dDistribution_gen(
    name="SkewNormal1dDistribution"
)


def skew_gaussian_1d(
    x, amplitude, center, width, alpha, offset=0.0
):
    r"""
    Skewed Gaussian in one dimension.

    See: https://en.wikipedia.org/wiki/Skew_normal_distribution.

    Parameters
    ----------
    x : `float`
        Variable.
    amplitude : `float`
        Amplitude of PDF.
    center : `float`
        Mode of PDF.
    width : `float`
        Standard deviation of PDF.
    alpha : `float`
        Parameter controlling skewness.
    offset : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    ns = SkewNormal1dDistribution
    # Change to standard skew normal parameterization
    mu = np.sqrt(2 / np.pi) * alpha / np.sqrt(1 + alpha**2)
    wx = width / np.sqrt(1 - mu**2)
    x0 = center - wx * ns.mode(alpha)
    # Calculate function
    res = amplitude * ns.pdf(x, alpha, loc=x0, scale=wx)
    amp = ns.amplitude(alpha, loc=x0, scale=wx)
    if np.isscalar(amp):
        if amp > 1e-50:
            res = res / amp
    else:
        np.divide(res, amp, out=res, where=(amp > 1e-50))
    return res + offset


class FitSkewGaussian1d(FitGaussian1d):

    """
    Fit class for :py:func:`skew_gaussian_1d`.

    Parameters
    ----------
    a : `float`
        amplitude
    x0 : `float`
        center
    wx : `float`
        width
    alpha : `float`
        parameter controlling skewness
    c : `float`
        offset
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitSkewGaussian1d")
    P_ALL = ["a", "x0", "wx", "alpha", "c"]
    P_DEFAULT = [1, 0, 1, 0, 0]
    DISTRIBUTION = SkewNormal1dDistribution

    @staticmethod
    def _func(var, *p):
        return skew_gaussian_1d(var, *p)

    def find_p0(self, *data):
        var_data, func_data, _ = self.split_fit_data(*data)
        # Find p0 from unskewed Gaussian
        a, x0, wx, c = self._find_p0_stat(var_data, func_data)
        # Extract peak data
        var_data = var_data[0]
        mask = (var_data > x0 - 2 * wx) & (var_data < x0 + 2 * wx)
        x, y = var_data[mask], func_data[mask]
        # If insufficient data
        if len(x) < 9:
            alpha = 0
        # Make probability mass function from peak data
        else:
            bin_edges = (x[1:] + x[:-1]) / 2
            bin_widths = bin_edges[1:] - bin_edges[:-1]
            bin_widths = np.concatenate([
                [bin_widths[0]], bin_widths, [bin_widths[-1]]
            ])
            pmf = y - c
            pmf[pmf < 0] = 0
            pmf = pmf * bin_widths
            pmf /= np.sum(pmf)
            # Estimate alpha from PMF skewness
            _mu1 = np.sum(x * pmf)
            _mu2 = np.sum((x - _mu1)**2 * pmf)
            _mu3 = np.sum((x - _mu1)**3 * pmf)
            skewness = _mu3 / _mu2**(3/2)
            # Check for invalid skewness (often due to few data points)
            if np.abs(skewness) > 0.8:
                alpha = 0
            else:
                alpha = SkewNormal1dDistribution.cv_skewness_to_alpha(skewness)
        self.p0 = [a, x0, wx, alpha, c]

    def find_popt(self, *args, **kwargs):
        # If alpha == 0, curve_fit has problems optimizing
        if "alpha" in self.pfit:
            alpha = self.get_p0()["alpha"]
            alpha_sign = 1 if alpha == 0 else np.sign(alpha)
            if np.abs(alpha) < 1e-8:
                self.set_p0(alpha=1e-8*alpha_sign)
        return super().find_popt(*args, **kwargs)

    def get_distribution(self):
        """
        Gets the distribution corresponding to this model.

        Returns
        -------
        distr : `SkewNormal1dDistribution`
            Distribution object corresponding to best fit parameters.
        """
        center, width, alpha = self.x0, self.wx, self.alpha
        mu = np.sqrt(2 / np.pi) * alpha / np.sqrt(1 + alpha**2)
        wx = width / np.sqrt(1 - mu**2)
        x0 = center - wx * self.DISTRIBUTION.mode(alpha)
        return self.DISTRIBUTION(alpha, loc=x0, scale=wx)

    @property
    def distribution_amplitude(self):
        return self.a / self.get_distribution().amplitude()


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
    a : `float`
        amplitude
    x0, y0 : `float`
        center
    wu, wv : `float`
        width
    tilt : `float`
        tilt in [-45°, 45°]
    c : `float`
        offset
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitGaussian2dTilt")
    P_ALL = ["a", "x0", "y0", "wu", "wv", "tilt", "c"]
    P_DEFAULT = [1, 0, 0, 1, 1, 0, 0]

    @staticmethod
    def _func(var, *p):
        return gaussian_2d_tilt(var, *p)

    @staticmethod
    def _find_p0_linear(var_data, func_data):
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

    @staticmethod
    def _find_p0_fit1d(var_data, func_data):
        """
        Algorithm: 1D profile fits.
        """
        # Perform 1D profile fits
        len_x, len_y = func_data.shape
        xvar = var_data[0][..., 0]
        yvar = var_data[1][0]
        xdata = np.sum(func_data, axis=1)
        ydata = np.sum(func_data, axis=0)
        fit_1d = FitGaussian1d()
        fit_1d.find_p0(xvar, xdata)
        fit_1d.find_popt(xvar, xdata)
        px = np.copy(fit_1d.popt)
        fit_1d.find_p0(yvar, ydata)
        fit_1d.find_popt(yvar, ydata)
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
        var_data, func_data, _ = self.split_fit_data(*data)
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
            psuccess &= np.all([
                getattr(self, f"{pname}_std") / getattr(self, pname) < 1
                for pname in ["a", "wu", "wv"]
            ])
        return psuccess

    @property
    def ellipticity(self):
        wu, wv = abs(self.wu), np.abs(self.wv)
        return np.abs(wu - wv) / max(wu, wv)


###############################################################################
# Polynomial Functions
###############################################################################


def parabolic_1d(x, a, x0, c=0):
    r"""
    Parabola in one dimension.

    .. math::
        A (x - x_0) + C

    Parameters
    ----------
    x : `float`
        Variable :math:`x`.
    a : `float`
        Amplitude :math:`A`.
    x0 : `float`
        Center :math:`x_0`.
    c : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    return a * (x - x0)**2 + c


class FitParabolic1d(ModelBase):

    """
    Fit class for :py:func:`parabolic_1d`.

    Parameters
    ----------
    a : `float`
        amplitude
    x0 : `float`
        center
    c  : `float`
        offset
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitParabolic1d")
    P_ALL = ["a", "x0", "c"]
    P_DEFAULT = [1, 0, 0]

    @staticmethod
    def _func(var, *p):
        return parabolic_1d(var, *p)

    def find_p0(self, *data):
        var_data, func_data, _ = self.split_fit_data(*data)
        x, y = var_data[0], func_data
        # Smoothen data
        if len(y) < 5:
            _y_savgol = y
        else:
            _y_savgol = signal.savgol_filter(y, max(5, 2*(len(y)//5) + 1), 3)
        # Find polarity via peak and edges
        _y_med = np.median(_y_savgol)
        _xl, _xr = x[0], x[-1]
        _yl, _yr = _y_savgol[0], _y_savgol[-1]
        _sgn = 1 if np.mean([_yl, _yr]) > _y_med else -1
        if _sgn > 0:
            _cidx = np.argmin(_y_savgol)
        else:
            _cidx = np.argmax(_y_savgol)
        # Find p0
        _x0 = x[_cidx]
        _c = _y_savgol[_cidx]
        _a = np.mean([
            (_yl - _c) / (_xl - _x0)**2,
            (_yr - _c) / (_xr - _x0)**2,
        ])
        self.p0 = [_a, _x0, _c]


def parabolic_1d_finsup(x, a, x0, wx, c=0):
    r"""
    Parabola on finite support in one dimension.

    .. math::
        \left(a - \frac{\sign{a}}{2} \left(\frac{x - x_0}{w_x}\right)^2\right)
        \Theta (w_x \sqrt{2 A} - |x - x_0|) + c

    Parameters
    ----------
    x : `float`
        Variable :math:`x`.
    a : `float`
        Amplitude :math:`A`.
    x0 : `float`
        Center :math:`x_0`.
    wx : `float`
        Width of parabola :math:`w_x`.
        Normalized to be consistent with Gaussian width in series expansion.
    c : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    dx = x - x0
    return (
        c + (a - (dx / wx)**2 / 2 * np.sign(a))
        * np.sign(wx * np.sqrt(2 * np.abs(a)) - np.abs(dx))
    )


def parabolic_1d_int2d(var, a, x0, wx, c=0):
    """
    3D parabola on finite support integrated over two dimensions.
    """
    arg = 1 - (var - x0)**2 / wx**2
    arg[arg * np.sign(a) < 0] = 0
    res = a * arg**2 + c
    return res


class FitParabolic1dInt2d(FitGaussian1d):

    """
    Fit class for :py:func:`parabolic_1d_int2d`.

    Parameters
    ----------
    a : `float`
        amplitude
    x0 : `float`
        center
    wx : `float`
        width
    c : `float`
        offset
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitParabolic1dInt2d")
    P_ALL = ["a", "x0", "wx", "c"]
    P_DEFAULT = [1, 0, 1, 0]

    @staticmethod
    def _func(var, *p):
        return parabolic_1d_int2d(var, *p)

    def find_p0(self, *data):
        super().find_p0(*data)
        self.p0[self.pall["wx"]] *= np.sqrt(2)


def parabolic_2d_int1d_tilt(var, a, x0, y0, wu, wv, tilt=0, c=0):
    """
    3D parabola on finite support integrated over one dimension.
    """
    tilt_cos, tilt_sin = np.cos(tilt), np.sin(tilt)
    dx, dy = var[0] - x0, var[1] - y0
    arg = (
        1 - (dx * tilt_cos + dy * tilt_sin)**2 / wu**2
        - (dy * tilt_cos - dx * tilt_sin)**2 / wv**2
    )
    arg[arg * np.sign(a) < 0] = 0
    res = a * arg**(3/2) + c
    return res


class FitParabolic2dInt1dTilt(FitGaussian2dTilt):

    """
    Fit class for :py:func:`parabolic_1d_int2d`.

    Parameters
    ----------
    a : `float`
        amplitude
    x0, y0 : `float`
        center
    wu, wv : `float`
        width
    tilt : `float`
        tilt in [-45°, 45°]
    c : `float`
        offset
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitParabolic2dInt1dTilt")
    P_ALL = ["a", "x0", "y0", "wu", "wv", "tilt", "c"]
    P_DEFAULT = [1, 0, 0, 1, 1, 0, 0]

    @staticmethod
    def _func(var, *p):
        return parabolic_2d_int1d_tilt(var, *p)

    def _find_p0_linear(self, *data):
        p0 = super()._find_p0_linear(*data)
        for key in ["wu", "wv"]:
            p0[self.pall[key]] *= np.sqrt(2)
        return p0

    @staticmethod
    def _find_p0_fit1d(var_data, func_data):
        # Perform 1D profile fits
        len_x, len_y = func_data.shape
        xvar = var_data[0][..., 0]
        yvar = var_data[1][0]
        xdata = np.sum(func_data, axis=1)
        ydata = np.sum(func_data, axis=0)
        fit_1d = FitParabolic1dInt2d()
        fit_1d.find_p0(xvar, xdata)
        fit_1d.find_popt(xvar, xdata)
        px = np.copy(fit_1d.popt)
        fit_1d.find_p0(yvar, ydata)
        fit_1d.find_popt(yvar, ydata)
        py = np.copy(fit_1d.popt)
        # Use 1D fit parameters for 2D fit initial parameters
        ax, x0, wx, cx = px
        ay, y0, wy, cy = py
        a = np.mean([ax / wy, ay / wx]) / (3/8 * np.pi)
        tilt = 0
        c = np.mean([cx / len_y, cy / len_x])
        return [a, x0, y0, wx, wy, tilt, c]


###############################################################################
# Bimodal Gaussian Functions
###############################################################################


def bm_gaussian_parabolic_1d_int2d(x, ag, ap, x0, wgx, wpx, c=0):
    g = gaussian_1d(x, ag, x0, wgx)
    p = parabolic_1d_int2d(x, ap, x0, wpx)
    return g + p + c


class FitBmGaussianParabolic1dInt2d(FitGaussian1d):

    """
    Fit class for :py:func:`bm_gaussian_parabolic_1d_int2d`.

    Parameters
    ----------
    ag, ap : `float`
        amplitudes of Gaussian and parabola
    x0 : `float`
        center
    wgx, wpx : `float`
        width of Gaussian and parabola
    c : `float`
        offset
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitBmGaussianParabolic1d")
    P_ALL = ["ag", "ap", "x0", "wgx", "wpx", "c"]
    P_DEFAULT = [0.5, 0.5, 0, 1, 0.5, 0]

    @staticmethod
    def _func(var, *p):
        return bm_gaussian_parabolic_1d_int2d(var, *p)

    def find_p0(self, *data):
        """
        Algorithm: find_p0 of Gaussian -> split data at width
        -> fit tails with Gaussian, center with parabola.

        Notes
        -----
        Currently only works for positive data.
        """
        var_data, func_data, _ = self.split_fit_data(*data)
        fitg, fitp = FitGaussian1d(), FitParabolic1dInt2d()
        # Split data
        fitg.find_p0(var_data, func_data)
        _x0, _wx = fitg.p0[[1, 2]]
        _var_mask = (np.abs(var_data - _x0) > _wx)
        _fun_mask = _var_mask.ravel()
        _var_data_g, _func_data_g = var_data[_var_mask], func_data[_fun_mask]
        _var_data_p, _func_data_p = var_data[~_var_mask], func_data[~_fun_mask]
        # Fit split data
        fitg.find_p0(_var_data_g, _func_data_g)
        fitg.find_popt(_var_data_g, _func_data_g)
        ag, x0g, wgx, cg = fitg.popt
        fitp.find_p0(_var_data_p, _func_data_p)
        fitp.find_popt(_var_data_p, _func_data_p)
        ap, x0p, wpx, cp = fitp.popt
        # Assign p0
        x0 = np.mean([x0g, x0p])
        c = 0.8 * cg + 0.2 * cp
        self.p0 = [ag, ap, x0, wgx, wpx, c]

    def get_fit_gaussian_1d(self):
        """
        Gets a fit object only modelling the Gaussian part.
        """
        fit = FitGaussian1d()
        fit.p0 = [self.ag, self.x0, self.wgx, self.c]
        return fit

    def get_fit_parabolic_1d_int2d(self):
        """
        Gets a fit object only modelling the parabolic part.
        """
        fit = FitParabolic1dInt2d()
        fit.p0 = [self.ap, self.x0, self.wpx, 0]
        return fit


def bm_gaussian_parabolic_2d_int1d(
    var, ag, ap, x0, y0, wgu, wgv, wpu, wpv, tilt=0, c=0
):
    g = gaussian_2d_tilt(var, ag, x0, y0, wgu, wgv, tilt)
    p = parabolic_2d_int1d_tilt(var, ap, x0, y0, wpu, wpv, tilt)
    return g + p + c


class FitBmGaussianParabolic2dInt1dTilt(FitGaussian2dTilt):

    """
    Fit class for :py:func:`bm_gaussian_parabolic_2d_int1d`.

    Parameters
    ----------
    ag, ap : `float`
        amplitudes of Gaussian and parabola
    x0, y0 : `float`
        center
    wgu, wgv, wpu, wpv : `float`
        widths of Gaussian and parabola
    tilt : `float`
        tilt in [-45°, 45°]
    c : `float`
        offset
    """

    LOGGER = logging.get_logger(
        "libics.math.peaked.FitBmGaussianParabolic2dInt1dTilt"
    )
    P_ALL = ["ag", "ap", "x0", "y0", "wgu", "wgv", "wpu", "wpv", "tilt", "c"]
    P_DEFAULT = [0.5, 0.5, 0, 0, 1, 1, 0.5, 0.5, 0, 0]

    @staticmethod
    def _func(var, *p):
        return bm_gaussian_parabolic_2d_int1d(var, *p)

    @staticmethod
    def _find_p0_fit1d(var_data, func_data):
        # Perform 1D profile fits
        len_x, len_y = func_data.shape
        xvar = var_data[0][..., 0]
        yvar = var_data[1][0]
        xdata = np.sum(func_data, axis=1)
        ydata = np.sum(func_data, axis=0)
        fit_1d = FitBmGaussianParabolic1dInt2d()
        fit_1d.find_p0(xvar, xdata)
        fit_1d.find_popt(xvar, xdata)
        px = np.copy(fit_1d.popt)
        fit_1d.find_p0(yvar, ydata)
        fit_1d.find_popt(yvar, ydata)
        py = np.copy(fit_1d.popt)
        # Use 1D fit parameters for 2D fit initial parameters
        agx, apx, x0, wgx, wpx, cx = px
        agy, apy, y0, wgy, wpy, cy = py
        ag = np.mean([agx / wgy, agy / wgx]) / np.sqrt(2 * np.pi)
        ap = np.mean([apx / wpy, apy / wpx]) / (3/8 * np.pi)
        tilt = 0
        c = np.mean([cx / len_y, cy / len_x])
        return [ag, ap, x0, y0, wgx, wgy, wpx, wpy, tilt, c]

    def find_p0(self, *data):
        var_data, func_data, _ = self.split_fit_data(*data)
        self.p0 = self._find_p0_fit1d(var_data, func_data)

    def find_popt(self, *args, **kwargs):
        psuccess = super().find_popt(*args, **kwargs)
        if psuccess:
            # Enforce positive widths
            for pname in ["wgu", "wgv", "wpu", "wpv"]:
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
                    if np.all((pname in self.pfit
                               for pname in ["wgu", "wgv", "wpu", "wpv"])):
                        tilt -= 1/2 * np.pi
                        wgu_idx, wgv_idx = self.pfit["wgu"], self.pfit["wpv"]
                        wpu_idx, wpv_idx = self.pfit["wpu"], self.pfit["wpv"]
                        popt = self.popt_for_fit
                        pcov = self.pcov_for_fit
                        popt[[wgu_idx, wgv_idx]] = popt[[wgv_idx, wgu_idx]]
                        popt[[wpu_idx, wpv_idx]] = popt[[wpv_idx, wpu_idx]]
                        pcov[[wgu_idx, wgv_idx]] = pcov[[wgu_idx, wgv_idx]]
                        pcov[[wpu_idx, wpv_idx]] = pcov[[wpu_idx, wpv_idx]]
                        pcov[:, [wgu_idx, wgv_idx]] = (
                            pcov[:, [wgu_idx, wgv_idx]]
                        )
                        pcov[:, [wpu_idx, wpv_idx]] = (
                            pcov[:, [wpu_idx, wpv_idx]]
                        )
                        self.popt_for_fit = popt
                        self.pcov_for_fit = pcov
                self.popt_for_fit[tilt_idx] = tilt
        return psuccess

    def get_fit_gaussian_2d_tilt(self):
        """
        Gets a fit object only modelling the Gaussian part.
        """
        fit = FitGaussian2dTilt()
        fit.p0 = [self.ag, self.x0, self.y0, self.wgu, self.wgv,
                  self.tilt, self.c]
        return fit

    def get_fit_parabolic_2d_int1d_tilt(self):
        """
        Gets a fit object only modelling the parabolic part.
        """
        fit = FitParabolic2dInt1dTilt()
        fit.p0 = [self.ap, self.x0, self.y0, self.wpu, self.wpv,
                  self.tilt, 0]
        return fit


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
    a : `float`
        amplitude
    x0 : `float`
        center
    wx : `float`
        width
    c : `float`
        offset
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
        var_data, func_data, _ = self.split_fit_data(*data)
        var_data = var_data[0]
        idx_max = np.argmax(func_data)
        a = func_data[idx_max]
        x0 = var_data[idx_max]
        wx = (np.max(var_data) - np.min(var_data)) / 4
        c = 0
        self.p0 = [a, x0, wx, c]

    def find_popt(self, *args, **kwargs):
        psuccess = super().find_popt(*args, **kwargs)
        if psuccess:
            # Enforce positive width
            for pname in ["wx"]:
                if pname in self.pfit:
                    pidx = self.pfit[pname]
                    self._popt[pidx] = np.abs(self._popt[pidx])
        return psuccess


def lorentzian_eit_1d_imag(
    var, amplitude, center, width, shift, split, offset=0.0
):
    dx = var - center
    denom = np.piecewise(
        dx, [dx == shift],
        [lambda _dx: np.inf,
         lambda _dx: 1 + (_dx / width - split / (_dx - shift))**2]
    )
    return amplitude / denom + offset


class FitLorentzianEit1dImag(ModelBase):

    """
    Fit class for :py:func:`lorentzian_eit_1d_imag`.

    Parameters
    ----------
    a : `float`
        amplitude
    x0 : `float`
        center
    wx : `float`
        width
    x1 : `float`
        shift
    wx1 : `float`
        split
    c : `float`
        offset

    Attributes
    ----------
    ge : `float`
        excited state decay rate
    fc : `float`
        control field Rabi frequency
    dc : `float`
        control field detuning
    lmax : `float`
        position of left maximum
    rmax : `float`
        position of right maximum
    cmin : `float`
        position of central minimum
    lwidth, rwidth : `float`
        width of left/right maximum
    cwidth : `float`
        width of central minimum
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitLorentzianEit1dImag")
    P_ALL = ["a", "x0", "wx", "x1", "wx1", "c"]
    P_DEFAULT = [1, 0, 1, 0, 1, 0]

    @staticmethod
    def _func(var, *p):
        return lorentzian_eit_1d_imag(var, *p)

    @property
    def ge(self):
        return np.abs(self.wx)

    @property
    def fc(self):
        return 2 * np.sqrt(np.abs(self.wx * self.wx1))

    @property
    def dp(self):
        return self.x0

    @property
    def dc(self):
        return -self.x1

    def get_phys(self):
        """
        Gets parameters parametrized as physical quantities.

        Returns
        -------
        phys : `dict(str->float)`
            Parameters: amplitude `"a"`, probe resonance frequency `"dp"`,
            control resonance frequency `"dc"`, control Rabi frequency: `"fc"`,
            intermediate state decay rate "ge", offset `"c"`.
        """
        return {"a": self.a, "dp": self.dp, "dc": self.dc,
                "fc": self.fc, "ge": self.ge, "c": self.c}

    @property
    def lmax(self):
        return self.x1 / 2 - np.sqrt(
            self.x1**2 / 4 + np.abs(self.wx * self.wx1)
        ) + self.x0

    @property
    def rmax(self):
        return self.x1 / 2 + np.sqrt(
            self.x1**2 / 4 + np.abs(self.wx * self.wx1)
        ) + self.x0

    @property
    def cmin(self):
        return self.x1 + self.x0

    @property
    def lwidth(self):
        _xr = self.rmax
        return (self.wx * _xr**2 / np.abs(self.x1 - 2*_xr)
                / np.sqrt(self.wx*self.wx1 + self.x1*_xr))

    @property
    def rwidth(self):
        _xl = self.lmax
        return (self.wx * _xl**2 / np.abs(self.x1 - 2*_xl)
                / np.sqrt(self.wx*self.wx1 + self.x1*_xl))

    @property
    def cwidth(self):
        return np.abs(self.wx1)

    # +++++++++++++++++++++++++++++++++++++++++++++++

    def find_p0(self, *data):
        var_data, func_data, _ = self.split_fit_data(*data)
        x, y_noisy = var_data[0], func_data
        # Smoothen data
        y_savgol = signal.savgol_filter(
            y_noisy, max(5, 2*(len(y_noisy)//16//2) + 1), 3
        )
        _ymin = np.percentile(y_savgol, 100*(1/16))
        _ymax = np.percentile(y_savgol, 100*(1-1/32))
        _a, _c = _ymax - _ymin, _ymin
        # Peak detection (maxima)
        _y_rescaled = (y_savgol - _c) / _a
        _peak_max_idx, _peak_max_prominence = signal.find_peaks(
            _y_rescaled, prominence=0.1
        )
        _order = np.flip(np.argsort(_peak_max_prominence["prominences"]))[:2]
        _peak_max_idx = np.sort(_peak_max_idx[_order])
        _peak_max_pos = x[_peak_max_idx]
        # Peak detection (minimum)
        _peak_min_idx, _peak_min_prominence = signal.find_peaks(
            -_y_rescaled, prominence=0.1
        )
        _order = np.argmax(_peak_min_prominence["prominences"])
        _peak_min_idx = _peak_min_idx[_order]
        _peak_min_pos = x[_peak_min_idx]
        # Estimation of x parameters
        _xl, _xc, _xr = _peak_max_pos[0], _peak_min_pos, _peak_max_pos[1]
        _x0 = _xl + _xr - _xc
        _x1 = _xc - _x0
        _w0w1 = 1/4 * np.abs((_xl - _xr)**2 - _x1**2)
        # Fit parabola to widest peak
        _dxl, _dxr = _xc - _xl, _xr - _xc
        if _dxl > _dxr:
            _xpeak = _xl
            _dst = max(_dxr, _dxl / 5)
        else:
            _xpeak = _xr
            _dst = max(_dxl, _dxr / 5)
        _idx_range = slice(
            np.argmin(np.abs(x - (_xpeak - _dst))),
            np.argmin(np.abs(x - (_xpeak + _dst))) + 1
        )
        _fit = FitParabolic1d(x[_idx_range], y_savgol[_idx_range])
        _peak_width = 1 / np.sqrt(-_fit.a)
        # Estimation of w parameters
        _w0 = (_peak_width / (_xpeak - _x0)**2 * np.abs(_x1 - 2*(_xpeak - _x0))
               * np.sqrt(np.abs(_w0w1 + _x1*(_xpeak - _x0))))
        _w1 = _w0w1 / _w0
        self.p0 = [_a, _x0, _w0, _x1, _w1, _c]

    def find_popt(self, *args, **kwargs):
        psuccess = super().find_popt(*args, **kwargs)
        if psuccess:
            # Enforce positive width
            for pname in ["wx", "wx1"]:
                if pname in self.pfit:
                    pidx = self.pfit[pname]
                    self._popt[pidx] = np.abs(self._popt[pidx])
        return psuccess


def lorentzian_ryd_eit_1d_imag(
    var, amplitude, center, width, shift, split, ratio, offset=0.0
):
    """
    See :py:func:`lorentzian_eit_1d_imag`.

    Parameters
    ----------
    ratio : `float`
        Rydberg fraction (ratio of non-EIT).
    """
    return (
        (1 - ratio) * lorentzian_eit_1d_imag(
            var, amplitude, center, width, shift, split
        ) + ratio * lorentzian_1d_abs(
            var, amplitude, center, width,
        ) + offset
    )


class FitLorentzianRydEit1dImag(FitLorentzianEit1dImag):

    """
    Fit class for :py:func:`lorentzian_ryd_eit_1d_imag`.

    Parameters
    ----------
    a : `float`
        amplitude
    x0 : `float`
        center
    wx : `float`
        width
    x1 : `float`
        shift
    wx1 : `float`
        split
    r : `float`
        ratio
    c : `float`
        offset

    Attributes
    ----------
    ge : `float`
        excited state decay rate
    fc : `float`
        control field Rabi frequency
    dc : `float`
        control field detuning
    lmax, rmax : `float`
        position of left/right maximum
    cmin : `float`
        position of central minimum
    lwidth, rwidth : `float`
        width of left/right maximum
    cwidth : `float`
        width of central minimum
    """

    LOGGER = logging.get_logger("libics.math.peaked.FitLorentzianRydEit1dImag")
    P_ALL = ["a", "x0", "wx", "x1", "wx1", "r", "c"]
    P_DEFAULT = [1, 0, 1, 0, 1, 0.5, 0]

    @staticmethod
    def _func(var, *p):
        return lorentzian_ryd_eit_1d_imag(var, *p)

    def find_p0(self, *data, const_p0=None):
        """
        Parameters
        ----------
        const_p0 : `dict(str->float)`
            Constant parameters which should not be fitted when estimating
            `p0` with `r == 0` fit.
        """
        var_data, func_data, _ = self.split_fit_data(*data)
        x, y = var_data[0], func_data
        # Find raw EIT
        _fit = FitLorentzianEit1dImag()
        _fit.find_p0(*data)
        if const_p0:
            _fit.set_pfit(**const_p0)
        _fit.find_popt(*data)
        _p0 = dict()
        if const_p0:
            _p0.update(const_p0)
        _p0.update({k: _fit.popt[v] for k, v in _fit.pfit.items()})
        # Fit central peak
        _xc = _fit.cmin
        _wc = _fit.cwidth
        _xc_range = [_xc - _wc, _xc + _wc]
        _cidx = [
            np.argmin(np.abs(x - _xc_range[0])),
            np.argmin(np.abs(x - _xc_range[1])) + 1
        ]
        if _cidx[1] - _cidx[0] < 5:
            _cidx = [_cidx[0] - 2, _cidx[1] + 2]
        if _cidx[0] < 0:
            _cidx[0] = 0
        _cidx_range = slice(*_cidx)
        _fit_parabola = FitParabolic1d(x[_cidx_range], y[_cidx_range])
        # Find ratio
        _r = 1 - (_fit_parabola.c - _p0["c"]) / _p0["a"]
        if _r > 1:
            _r = 1
        elif _r < 0:
            _r = 0
        _p0["r"] = _r
        self.set_p0(**_p0)


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
    var : `float`
        :math:`(x, y)`
    amplitude : `float`
        :math:`A`
    center_x, center_y : `float`
        :math:`(x_0, y_0)`
    width : `float`
        :math:`w`
    offset : `float`
        :math:`C`

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
    a : `float`
        amplitude
    x0, y0 : `float`
        center
    w : `float`
        width
    c : `float`
        offset
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
        var_data, func_data, _ = self.split_fit_data(*data)
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
    var : `float`
        :math:`(n, t)`
    hopping : `float`
        :math:`J`
    gradient : `float`
        :math:`\Delta`

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

    Parameters
    ----------
    var : `float`
        :math:`(n_x, n_y, t)`
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

    Parameters
    ----------
    var : `float`
        :math:`(n_x, n_y, n_z, t)`
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
    var : `float`
        :math:`(n, t)`
    hopping : `float`
        :math:`J`

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
    var : `float`
        :math:`(n, t)`
    diffusion : `float`
        :math:`D`

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
