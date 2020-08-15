import colorsys
import numpy as np


###############################################################################


def change_brightness_rgb(rgb, scale):
    """
    Scales the brightness of a RGB color in HLS space.
    """
    hls = np.array(rgb_to_hls(rgb))
    l_old = hls[1]
    hls[1] *= scale
    hls[1] = max(0.13 * l_old, min(0.87 + 0.13 * l_old, hls[1]))
    rgb = hls_to_rgb(hls)
    return rgb


def darken_rgb(*rgbs, scale=0.5):
    size = len(rgbs)
    rgbs = np.array([change_brightness_rgb(rgb, scale) for rgb in rgbs])
    if size == 1:
        return rgbs[0]


def lighten_rgb(*rgbs, scale=2):
    size = len(rgbs)
    rgbs = np.array([change_brightness_rgb(rgb, scale) for rgb in rgbs])
    if size == 1:
        return rgbs[0]


def interpolate_rgb(rgb1, rgb2, scale=0.5):
    """
    Interpolates linearly between two RGB colors in HLS space.

    Parameters
    ----------
    rgb1, rgb2
        RGB colors.
    scale : `float`
        Relative interpolation position between the two colors,
        where `(0, 1)` corresponds to `(rgb1, rgb2)`.
    """
    hls1, hls2 = [np.array(rgb_to_hls(rgb)) for rgb in (rgb1, rgb2)]
    hls = scale * hls1 + (1 - scale) * hls2
    return hls_to_rgb(hls)


###############################################################################
# Conversions
###############################################################################


def hex_to_rgb(hex):
    if isinstance(hex, str):
        rgb = tuple(int(hex.strip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
    else:
        rgb = hex
    return rgb


def rgb_to_hls(rgb):
    rgb, a = _remove_alpha(rgb)
    hls = colorsys.rgb_to_hls(*rgb)
    return _add_alpha(hls, a)


def hls_to_rgb(hls):
    hls, a = _remove_alpha(hls)
    rgb = colorsys.hls_to_rgb(*hls)
    return _add_alpha(rgb, a)


def rgb_to_hsv(rgb):
    rgb, a = _remove_alpha(rgb)
    hsv = colorsys.rgb_to_hsv(*rgb)
    return _add_alpha(hsv, a)


def hsv_to_rgb(hsv):
    hsv, a = _remove_alpha(hsv)
    rgb = colorsys.hls_to_rgb(*hsv)
    return _add_alpha(rgb, a)


def normalize(ar, vmin=None, vmax=None):
    """
    Normalizes an array linearly onto the [0, 1] interval.

    Parameters
    ----------
    ar : `np.ndarray(float)`
        Array-like to be normalized.
    vmin, vmax : `float`
        Normalization interval on the current scale.
        If `None`, uses the array minimum and maximum, respectively.
        If `vmin` equals `vmax`, a zero-array is returned.

    Returns
    -------
    norm : `np.ndarray(float)`
        Normalized array.
    """
    ar = np.array(ar)
    if vmin is None:
        vmin = np.min(ar)
    if vmax is None:
        vmax = np.max(ar)
    if vmin >= vmax:
        return np.zeros_like(ar)
    else:
        return (ar - vmin) / (vmax - vmin)


###############################################################################
# Helper functions
###############################################################################


def _remove_alpha(color):
    """
    Parameters
    ----------
    color : `3- or 4-tuple(float)`
        Normalized color.

    Returns
    -------
    color : `3-tuple(float)`
        Color without alpha channel.
    alpha : `float` or `None`
        Alpha channel value or `None`.
    """
    alpha = None
    if len(color) == 4:
        alpha = color[-1]
        color = color[:3]
    return color, alpha


def _add_alpha(color, alpha):
    """
    Inverse function to :py:func:`_remove_alpha`.
    """
    if alpha is not None:
        color = (color[0], color[1], color[2], alpha)
    return color
