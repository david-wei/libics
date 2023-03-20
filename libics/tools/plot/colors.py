from colorspacious import cspace_convert, deltaE
import colorsys
from functools import lru_cache, wraps
import matplotlib as mpl
import numpy as np
import scipy.optimize


###############################################################################
# DEPRECATED (based on HLS color space, use CIECAM02 functions instead)
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


def hex_to_rgb(hex):
    if isinstance(hex, str):
        rgb = tuple(int(hex.strip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
    else:
        rgb = hex
    return rgb


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
# Matplotlib functions
###############################################################################

class _CMAP_COUNTER:
    val = 0


def _assume_cmap(cmap):
    if isinstance(cmap, (mpl.colors.Colormap, str)):
        cmap = mpl.colormaps.get_cmap(cmap)
    else:
        cmap = make_cmap(cmap)
    return cmap


def add_named_color(name, color):
    """Add a named RGB(A) color to matplotlib."""
    if isinstance(color, str) and color[0] != "#":
        color = "#" + color
    color_hex = mpl.colors.to_hex(color)
    mpl.colors.get_named_colors_mapping()[name] = color_hex


def add_named_cmap(cmap, name=None, **kwargs):
    """Add a named color map to matplotlib."""
    if not isinstance(cmap, mpl.colors.Colormap):
        cmap = make_cmap(cmap, name=name, **kwargs)
    mpl.colormaps.register(cmap, name=name)


def make_cmap(colors, name=None, continuous=True):
    """Creates a matplotlib continuous or discrete colormap."""
    if name is None:
        name = f"custom_cmap{_CMAP_COUNTER.val:d}"
        _CMAP_COUNTER.val += 1
    if continuous:
        cmap = mpl.colors.LinearSegmentedColormap.from_list(name, colors)
    else:
        cmap = mpl.colors.ListedColormap(colors, name=name)
    return cmap


def get_colors_from_cmap(cmap, num=6):
    """Gets discrete colors from a colormap."""
    cmap = _assume_cmap(cmap)
    return cmap(np.linspace(0, 1, num=num))


def get_color_from_cmap(cmap, scale=0):
    """Gets a color at the position `scale` (within [0, 1]) from a colormap."""
    cmap = _assume_cmap(cmap)
    return cmap(np.array([0, scale]))[1]


def set_color_cycle(colors):
    """Change the matplotlib default color cycle to the given colors."""
    color_cycler = mpl.cycler(color=colors)
    mpl.rcParams["axes.prop_cycle"] = color_cycler


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


def _srgb_parser(multi=False, squeeze=False):
    """
    Returns a decorator for parsing color strings and handling alpha.

    Parameters
    ----------
    multi : `bool`
        Whether the decorated function accepts multiple colors as `*args`.
    squeeze : `bool`
        If `multi is True`, sets whether a scalar color is returned
        if the multi-color iterable has only one element.
    """

    if multi is False:
        def _decorator(function):
            def parsing_function(color, *args, cv_out=None, **kwargs):
                if isinstance(color, str):
                    color = mpl.colors.to_rgba(color)
                rgb_color, alpha = _remove_alpha(color)
                rgb_color = np.array(function(rgb_color, *args, **kwargs))
                if rgb_color.ndim == 1:
                    color = np.array(_add_alpha(rgb_color, alpha))
                    if cv_out is not None:
                        color = cv_out(color)
                else:
                    color = np.array([
                        _add_alpha(_rgb_color, alpha)
                        for _rgb_color in rgb_color
                    ])
                    if cv_out is not None:
                        color = cv_out(*color)
                return color
            parsing_function.__doc__ = function.__doc__
            return parsing_function

    elif squeeze is False:
        def _decorator(function):
            def parsing_function(*colors, cv_out=None, **kwargs):
                rgb_colors, alphas = [], []
                for color in colors:
                    if isinstance(color, str):
                        color = mpl.colors.to_rgba(color)
                    rgb_color, alpha = _remove_alpha(color)
                    rgb_colors.append(rgb_color)
                    alphas.append(alpha)
                rgb_colors = function(*rgb_colors, **kwargs)
                colors = np.array([
                    _add_alpha(rgb_color, alphas[i % len(alphas)])
                    for i, rgb_color in enumerate(rgb_colors)
                ])
                if cv_out is not None:
                    colors = cv_out(colors)
                return colors
            parsing_function.__doc__ = function.__doc__
            return parsing_function

    else:
        def _decorator(function):
            def parsing_function(*colors, cv_out=None, **kwargs):
                rgb_colors, alphas = [], []
                for color in colors:
                    if isinstance(color, str):
                        color = mpl.colors.to_rgba(color)
                    rgb_color, alpha = _remove_alpha(color)
                    rgb_colors.append(rgb_color)
                    alphas.append(alpha)
                rgb_colors = function(*rgb_colors, **kwargs)
                colors = np.array([
                    _add_alpha(rgb_color, alphas[i % len(alphas)])
                    for i, rgb_color in enumerate(rgb_colors)
                ])
                if colors.ndim == 2 and len(colors) == 1:
                    colors = colors[0]
                if cv_out is not None:
                    colors = cv_out(colors)
                return colors
            parsing_function.__doc__ = function.__doc__
            return parsing_function

    return _decorator


def _color_lru_cache(function):
    """
    LRU cache supporting numpy arrays.

    See:
    https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays
    """

    @lru_cache(maxsize=1024)
    def cached_wrapper(hashable_array):
        array = np.array(hashable_array)
        return function(array)

    @wraps(function)
    def wrapper(array):
        return cached_wrapper(tuple(array))

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


###############################################################################
# CIECAM02 color space
###############################################################################

# +++++++++++++++++++++++++++++++++++++++++
# sRGB gamut calculations
# +++++++++++++++++++++++++++++++++++++++++

def _srgb_gamut_opt_func(val, curve_func):
    """
    Loss function for finding the sRGB gamut boundary.

    Optimization is performed along a curve in JCh space.

    Parameters
    ----------
    val : `float`
        Parameter for curve. Valid values are:
        lightness: [0, 100],
        chroma: [0, 100],
        saturation: [0°, 90°].
    curve_func : `Callable`
        Curve function with signature `curve_func(val)->color`.
    """
    if val < 0:
        return np.abs(val)
    new_jch = curve_func(val)
    new_rgb = cspace_convert(new_jch, "JCh", "sRGB1")
    loss_ofl = np.sum(new_rgb[new_rgb > 1] - 1)
    loss_ufl = np.sum(-new_rgb[new_rgb < 0])
    loss_reg = min(
        np.abs(np.max(new_rgb) - 1),
        np.abs(np.min(new_rgb))
    )
    return loss_ofl + loss_ufl + loss_reg


@_color_lru_cache
def _get_srgb_gamut_max_chroma(jch_color):
    new_jch = np.array(jch_color, dtype=float).copy()
    color_dim = 1
    curve_bounds = (0, 100)

    def curve_func(val):
        new_jch[color_dim] = val
        return new_jch

    res = scipy.optimize.minimize_scalar(
        _srgb_gamut_opt_func,
        args=(curve_func,),
        method="Bounded", bounds=curve_bounds
    )
    return res.x


@_color_lru_cache
def _get_srgb_gamut_max_lightness(jch_color):
    new_jch = np.array(jch_color, dtype=float).copy()
    color_dim = 0
    curve_bounds = (0, 100)

    def curve_func(val):
        new_jch[color_dim] = val
        return new_jch

    res = scipy.optimize.minimize_scalar(
        _srgb_gamut_opt_func,
        args=(curve_func,),
        method="Bounded", bounds=curve_bounds
    )
    return res.x


@_color_lru_cache
def _get_srgb_gamut_min_lightness(jch_color):
    new_jch = np.array(jch_color, dtype=float).copy()
    color_dim = 0
    curve_bounds = (0, 100)

    def curve_func(val):
        new_jch[color_dim] = val
        return new_jch

    res = scipy.optimize.minimize_scalar(
        _srgb_gamut_opt_func,
        args=(curve_func,),
        method="Bounded", bounds=curve_bounds
    )
    return res.x


@_color_lru_cache
def _get_srgb_gamut_max_saturation(jch_color):
    new_jch = np.array(jch_color, dtype=float).copy()
    curve_bounds = (0, 90)

    def curve_func(val):
        _set_jch_saturation_brightness(new_jch, saturation_deg=val)
        return new_jch

    res = scipy.optimize.minimize_scalar(
        _srgb_gamut_opt_func,
        args=(curve_func,),
        method="Bounded", bounds=curve_bounds
    )
    return res.x


@_color_lru_cache
def _get_srgb_gamut_max_brightness(jch_color):
    new_jch = np.array(jch_color, dtype=float).copy()
    curve_bounds = (0, 100)

    def curve_func(val):
        _set_jch_saturation_brightness(new_jch, brightness=val)
        return new_jch

    res = scipy.optimize.minimize_scalar(
        _srgb_gamut_opt_func,
        args=(curve_func,),
        method="Bounded", bounds=curve_bounds
    )
    return res.x


def _set_jch_saturation_brightness(
    jch_color, saturation_deg=None, brightness=None
):
    """
    Changes the saturation and brightness of JCh color.

    Parameters
    ----------
    jch_color : `Array[n_color, float]`
        JCh color to be changed in-place.
    saturation_deg : `float` or `None`
        If `float`, sets the saturation in degrees.
    brightness : `float` or `None`
        If `float`, sets the brightness.
    """
    if saturation_deg is None:
        saturation_rad = np.arctan2(jch_color[1], jch_color[0])
    else:
        saturation_rad = np.deg2rad(saturation_deg)
    if brightness is None:
        brightness = np.linalg.norm(jch_color[:2])
    jch_color[0] = brightness * np.cos(saturation_rad)
    jch_color[1] = brightness * np.sin(saturation_rad)
    return jch_color


# +++++++++++++++++++++++++++++++++++++++++
# Change JCh color space parameters
# +++++++++++++++++++++++++++++++++++++++++


@_srgb_parser()
def rgb_change_chroma(rgb_color, chroma_change, scale="rel", clip_jch=True):
    """
    Changes the chroma of an sRGB color.

    Parameters
    ----------
    rgb_color : `Array[n_colors, float]`
        sRGB color to change.
    chroma_change : `float`
        Additive change of chroma.
    scale : `str`, optional
        Interprets `chroma_change` as follows:
        `"rel"`: relative to the in-gamut range.
        `"abs"`: as absolute value in JCh color space.
    clip_jch : `bool`, optional
        If the change results in an out-of-gamut color,
        whether to clip in JCh space instead of sRGB space.

    Returns
    -------
    rgb_color : `np.ndarray(n_colors, float)`
        Changed sRGB color.
    """
    jch_color = cspace_convert(rgb_color, "sRGB1", "JCh")
    if scale == "rel":
        chroma_change *= _get_srgb_gamut_max_chroma(jch_color)
    color_dim = 1
    jch_color[color_dim] += chroma_change
    jch_color[color_dim] = np.clip(jch_color[color_dim], 0, 100)
    if clip_jch:
        jch_color[color_dim] = np.clip(
            jch_color[color_dim],
            1,  # Using non-zero value to retain well-defined hue
            _get_srgb_gamut_max_chroma(jch_color)
        )
    rgb_color = cspace_convert(jch_color, "JCh", "sRGB1")
    rgb_color = np.clip(rgb_color, 0, 1)
    return rgb_color


@_srgb_parser()
def rgb_change_lightness(
    rgb_color, lightness_change, scale="rel", clip_jch=True
):
    """
    Changes the lightness of an sRGB color.

    Parameters
    ----------
    rgb_color : `Array[n_colors, float]`
        sRGB color to change.
    lightness_change : `float`
        Additive change of lightness.
    scale : `str`, optional
        Interprets `lightness_change` as follows:
        `"rel"`: relative to the in-gamut range.
        `"abs"`: as absolute value in JCh color space.
    clip_jch : `bool`, optional
        If the change results in an out-of-gamut color,
        whether to clip in JCh space instead of sRGB space.

    Returns
    -------
    rgb_color : `np.ndarray(n_colors, float)`
        Changed sRGB color.
    """
    jch_color = cspace_convert(rgb_color, "sRGB1", "JCh")
    if scale == "rel":
        lightness_change *= (
            _get_srgb_gamut_max_lightness(jch_color)
            - _get_srgb_gamut_min_lightness(jch_color)
        )
    color_dim = 0
    jch_color[color_dim] += lightness_change
    jch_color[color_dim] = np.clip(jch_color[color_dim], 0, 100)
    if clip_jch:
        jch_color[color_dim] = np.clip(
            jch_color[color_dim],
            _get_srgb_gamut_min_lightness(jch_color),
            _get_srgb_gamut_max_lightness(jch_color)
        )
    rgb_color = cspace_convert(jch_color, "JCh", "sRGB1")
    rgb_color = np.clip(rgb_color, 0, 1)
    return rgb_color


@_srgb_parser()
def rgb_change_saturation(
    rgb_color, saturation_change, scale="rel", clip_jch=True
):
    """
    Changes the saturation of an sRGB color.

    Parameters
    ----------
    rgb_color : `Array[n_colors, float]`
        sRGB color to change.
    saturation_change : `float`
        Additive change of lightness.
    scale : `str`, optional
        Interprets `lightness_change` as follows:
        `"rel"`: relative to the in-gamut range: [0, 1].
        `"abs"`: as absolute value in JCh color space: [0°, 90°].
    clip_jch : `bool`, optional
        If the change results in an out-of-gamut color,
        whether to clip in JCh space instead of sRGB space.

    Returns
    -------
    rgb_color : `np.ndarray(n_colors, float)`
        Changed sRGB color.
    """
    jch_color = cspace_convert(rgb_color, "sRGB1", "JCh")
    if scale == "rel":
        saturation_change *= _get_srgb_gamut_max_saturation(jch_color)
    brightness = np.linalg.norm(jch_color[:2])
    angle_deg = np.rad2deg(np.arctan2(jch_color[1], jch_color[0]))
    angle_deg += saturation_change
    angle_deg = np.clip(angle_deg, 0, 90)
    if clip_jch:
        angle_deg = np.clip(
            angle_deg,
            1,  # Using non-zero value to retain well-defined hue
            _get_srgb_gamut_max_saturation(jch_color)
        )
        _set_jch_saturation_brightness(jch_color, saturation_deg=angle_deg)
        brightness = np.clip(
            brightness,
            1,  # Using non-zero value to retain well-defined hue
            _get_srgb_gamut_max_brightness(jch_color)
        )
    _set_jch_saturation_brightness(
        jch_color, saturation_deg=angle_deg, brightness=brightness
    )
    rgb_color = cspace_convert(jch_color, "JCh", "sRGB1")
    rgb_color = np.clip(rgb_color, 0, 1)
    return rgb_color


@_srgb_parser()
def rgb_change_brightness(
    rgb_color, brightness_change, scale="rel", clip_jch=True
):
    """
    Changes the saturation of an sRGB color.

    Parameters
    ----------
    rgb_color : `Array[n_colors, float]`
        sRGB color to change.
    brightness_change : `float`
        Additive change of brightness.
    scale : `str`, optional
        Interprets `brightness_change` as follows:
        `"rel"`: relative to the in-gamut range: [0, 1].
        `"abs"`: as absolute value in JCh color space: [0, 100].
    clip_jch : `bool`, optional
        If the change results in an out-of-gamut color,
        whether to clip in JCh space instead of sRGB space.

    Returns
    -------
    rgb_color : `np.ndarray(n_colors, float)`
        Changed sRGB color.
    """
    jch_color = cspace_convert(rgb_color, "sRGB1", "JCh")
    if scale == "rel":
        brightness_change *= _get_srgb_gamut_max_brightness(jch_color)
    brightness = np.linalg.norm(jch_color[:2]) + brightness_change
    angle_deg = np.rad2deg(np.arctan2(jch_color[1], jch_color[0]))
    brightness = np.clip(brightness, 0, 100)
    if clip_jch:
        brightness = np.clip(
            brightness,
            1,  # Using non-zero value to retain well-defined hue
            _get_srgb_gamut_max_brightness(jch_color)
        )
        _set_jch_saturation_brightness(jch_color, brightness=brightness)
        angle_deg = np.clip(
            angle_deg,
            1,  # Using non-zero value to retain well-defined hue
            _get_srgb_gamut_max_saturation(jch_color)
        )
    _set_jch_saturation_brightness(
        jch_color, saturation_deg=angle_deg, brightness=brightness
    )
    _set_jch_saturation_brightness(jch_color, brightness=brightness)
    rgb_color = cspace_convert(jch_color, "JCh", "sRGB1")
    rgb_color = np.clip(rgb_color, 0, 1)
    return rgb_color


# +++++++++++++++++++++++++++++++++++++++++
# JCh color space interpolation functions
# +++++++++++++++++++++++++++++++++++++++++


@_srgb_parser()
def get_srgb_range(rgb_color, color_dim="lightness"):
    """
    Gets the smallest and largest value along a color dimension.

    Parameters
    ----------
    rgb_color : `Array[n_colors, float]`
        Reference sRGB color.
    color_dim : `str`, optional
        Color space dimension along which to return range:
        `"lightness", "chroma", "saturation"`

    Returns
    -------
    rgb_min, rgb_max : `tuple(np.ndarray(n_colors, float))`
        Minimum and maximum sRGB colors.
    """
    jch_color = cspace_convert(rgb_color, "sRGB1", "JCh")
    jch_colors = []
    if color_dim == "lightness":
        for lightness in (
            _get_srgb_gamut_min_lightness(jch_color),
            _get_srgb_gamut_max_lightness(jch_color)
        ):
            _jch_color = jch_color.copy()
            _jch_color[0] = lightness
            jch_colors.append(_jch_color)
    elif color_dim == "chroma":
        for chroma in (
            1,  # Using non-zero value to retain well-defined hue
            _get_srgb_gamut_max_chroma(jch_color)
        ):
            _jch_color = jch_color.copy()
            _jch_color[1] = chroma
            jch_colors.append(_jch_color)
    elif color_dim == "saturation":
        for saturation in (
            1,  # Using non-zero value to retain well-defined hue
            _get_srgb_gamut_max_saturation(jch_color)
        ):
            _jch_color = jch_color.copy()
            _set_jch_saturation_brightness(
                _jch_color, saturation_deg=saturation
            )
            jch_colors.append(_jch_color)
    else:
        raise ValueError(f"Invalid `color_dim`: {str(color_dim)}")
    rgb_colors = [
        cspace_convert(_jch_color, "JCh", "sRGB1")
        for _jch_color in jch_colors
    ]
    return np.clip(rgb_colors, 0, 1)


@_srgb_parser(multi=True)
def get_srgb_linspace(
    *keycolors, num=50, keycolor_distances=None, distance_measure="deltaE",
    hue_polarity=None
):
    """
    Returns an array of interpolated sRGB colors.

    Parameters
    ----------
    *keycolors : `Array[n_colors, float]`
        Keycolors as sRGB values.
    num : `int`, optional
        Number of interpolation points.
    keycolor_distances : `None` or `Array[1, float]`, optional
        Relative distances between the keycolors.
    distance_measure : `str`, optional
        `"deltaE"`: Scales `keycolor_distances` by the distance in JCh space.
        `"index"`: Does not scale `keycolor_distances`.
    hue_polarity : `str` or `None`
        As the hue is periodic, the interpolation can be performed
        `"clockwise"` or `"anticlockwise"`.
        If `None`, uses the polarity of the minimum hue distance.

    Returns
    -------
    rgb_interpolated_colors : `np.ndarray(2, float)`
        Interpolated sRGB colors with dimensions: `[num, n_colors]`
    """
    # Parse parameters
    if len(keycolors) == 0:
        keycolors = [np.zeros(3, dtype=float)]
    if len(keycolors) == 1:
        keycolors = get_srgb_range(keycolors[0])
    num_keycolors = len(keycolors)
    if keycolor_distances is None:
        keycolor_distances = np.full(
            num_keycolors - 1, 1 / (num_keycolors - 1), dtype=float
        )
    else:
        keycolor_distances = np.array(keycolor_distances, dtype=float)
    # Calculate keycolor locations
    if distance_measure == "deltaE":
        delta_e_distances = np.array([
            deltaE(keycolors[i], keycolors[i + 1])
            for i in range(num_keycolors - 1)
        ])
        keycolor_distances *= delta_e_distances
    keycolor_distances /= np.sum(keycolor_distances)
    num_per_distance = np.round(num * keycolor_distances).astype(int)
    excess_num = num - np.sum(num_per_distance)
    if excess_num > 0:
        idxs = np.argmax(num_per_distance)[:excess_num]
        for idx in idxs:
            num_per_distance[idx] -= 1
    # Transform keycolors to JCh
    jch_keycolors = cspace_convert(keycolors, "sRGB1", "JCh")
    # Handle hue periodicity
    for i in range(1, len(jch_keycolors)):
        hue_diff = jch_keycolors[i, 2] - jch_keycolors[i - 1, 2]
        if hue_polarity == "clockwise":
            if hue_diff < 0:
                jch_keycolors[i:, 2] += 360
        elif hue_polarity == "anticlockwise":
            if hue_diff > 0:
                jch_keycolors[i:, 2] -= 360
        else:
            if hue_diff > 180:
                jch_keycolors[i:, 2] -= 360
            elif hue_diff < -180:
                jch_keycolors[i:, 2] += 360
    # Interpolate colors
    jch_interpolated_colors = []
    for i in range(num_keycolors - 1):
        jch_start_color = jch_keycolors[i]
        jch_stop_color = jch_keycolors[i + 1]
        _num = num_per_distance[i]
        _interp = [
            np.linspace(jch_start_color[j], jch_stop_color[j], num=_num)
            for j in range(3)
        ]
        jch_interpolated_colors.append(np.transpose(_interp))
    jch_interpolated_colors = np.concatenate(jch_interpolated_colors)
    rgb_interpolated_colors = [
        cspace_convert(jch_color, "JCh", "sRGB1")
        for jch_color in jch_interpolated_colors
    ]
    return np.clip(rgb_interpolated_colors, 0, 1)


@_srgb_parser()
def get_srgb_gray_tinted(rgb_color_ref, grayscale):
    """
    Returns a grayscale sRGB color with infinitesimal chroma.

    Can be used to construct single-hue color interpolation.
    If pure grayscales would be used, a hue of 0° would be used.

    Parameters
    ----------
    rgb_color_ref : `Array[n_color, float]`
        RGB color defining the hue.
    grayscale : `float`
        Grayscale (clipped to the range [0.02, 0.98]).
        `0` corresponds to fully black, `1` to fully white.
    """
    grayscale = np.clip(grayscale, 0.02, 0.98)
    jch_color = cspace_convert(rgb_color_ref, "sRGB1", "JCh")
    jch_color[1] = 1
    jch_color[0] = grayscale * 100
    rgb_color = cspace_convert(jch_color, "JCh", "sRGB1")
    rgb_color = np.clip(rgb_color, 0, 1)
    return rgb_color


get_srgb_grey_tinted = get_srgb_gray_tinted


def get_srgb_white_tinted(rgb_color_ref):
    return get_srgb_gray_tinted(rgb_color_ref, 1)


def get_srgb_black_tinted(rgb_color_ref):
    return get_srgb_gray_tinted(rgb_color_ref, 0)


@_srgb_parser()
def rgb_whiten(rgb_color, scale=0.55):
    """
    Whitens a sRGB color by a given scale.

    Parameters
    ----------
    scale : `float`, optional
        `1` corresponds to fully white.
    """
    scale = np.clip(scale, 0, 1)
    jch_color = cspace_convert(rgb_color, "sRGB1", "JCh")
    jch_color[0] = scale * 100 + (1 - scale) * jch_color[0]
    jch_color[1] = (1 - scale) * jch_color[1]
    rgb_color = cspace_convert(jch_color, "JCh", "sRGB1")
    return np.clip(rgb_color, 0, 1)


@_srgb_parser()
def rgb_blacken(rgb_color, scale=0.55):
    """
    Blackens a sRGB color by a given scale.

    Parameters
    ----------
    scale : `float`, optional
        `1` corresponds to fully black.
    """
    scale = np.clip(scale, 0, 1)
    jch_color = cspace_convert(rgb_color, "sRGB1", "JCh")
    jch_color[:2] = (1 - scale) * jch_color[:2]
    rgb_color = cspace_convert(jch_color, "JCh", "sRGB1")
    return np.clip(rgb_color, 0, 1)


@_srgb_parser(multi=True, squeeze=True)
def rgb_equalize_lightness(*rgb_colors, trg_lightness=None, clip_jch=True):
    """
    Returns the RGB colors with equalized lightness, preserving saturation.

    Parameters
    ----------
    trg_lightness : `float` or `None` or sRGB color
        Target lightness in JCh color space (in range [0, 100]).
        If `None`, chooses a common lightness trying to
        preserve lightness of given colors.
        If sRGB color, uses its lightness as target.
    clip_jch : `bool``
        Whether to clip in JCh space (reduces chroma to fit in sRGB gamut).
    """
    jch_colors = cspace_convert(rgb_colors, "sRGB1", "JCh")
    lightnesses = jch_colors[..., 0]
    # Set target lightness from given colors
    if trg_lightness is None:
        trg_lightness = 50
        if np.all(lightnesses) > 50:
            trg_lightness = np.min(lightnesses)
        elif np.all(lightnesses) < 50:
            trg_lightness = np.max(lightnesses)
    # Set target lightness from target color
    elif not np.isscalar(trg_lightness) or isinstance(trg_lightness, str):
        trg_lightness = rgb_to_jch(trg_lightness)[0]
    # Equalize colors
    jch_colors[..., 1] *= trg_lightness / lightnesses  # Preserve saturation
    jch_colors[..., 0] = trg_lightness
    rgb_colors = cspace_convert(jch_colors, "JCh", "sRGB1")
    # Hue-preserving clipping
    if clip_jch:
        jch_max_chromas = [
            get_srgb_range(
                rgb_color, color_dim="chroma", cv_out=rgb_to_jch
            )[1][1] for rgb_color in rgb_colors
        ]
        mask = jch_colors[..., 1] > jch_max_chromas
        for i, clipped in enumerate(mask):
            if clipped:
                jch_colors[i, 1] = jch_max_chromas[i]
        rgb_colors = cspace_convert(jch_colors, "JCh", "sRGB1")
    return np.clip(rgb_colors, 0, 1)


###############################################################################
# Color space conversion functions
###############################################################################


@_srgb_parser(multi=True, squeeze=True)
def parse_color(*colors):
    """Parses color strings or hex codes to color values."""
    return colors


def rgb_to_hex(*rgb_colors):
    """Converts a RGB-style color to a hex string."""
    hex_colors = [mpl.colors.to_hex(rgb_color) for rgb_color in rgb_colors]
    return hex_colors


@_srgb_parser(multi=True, squeeze=True)
def rgb_to_jch(*rgb_colors):
    """Converts a RGB-style color to a JCh color."""
    jch_colors = cspace_convert(rgb_colors, "sRGB1", "JCh")
    return jch_colors


@_srgb_parser(multi=True, squeeze=True)
def jch_to_rgb(*jch_colors, clip=True):
    """Converts a RGB-style color to a JCh color."""
    rgb_colors = cspace_convert(jch_colors, "JCh", "sRGB1")
    if clip:
        rgb_colors = np.clip(rgb_colors, 0, 1)
    return rgb_colors


@_srgb_parser(multi=True, squeeze=True)
def rgb_to_hls(*rgb_colors):
    """Converts a RGB-style color to a HLS color."""
    return [colorsys.rgb_to_hls(*rgb) for rgb in rgb_colors]


def hls_to_rgb(hls):
    """Converts a HLS color to a RGB color."""
    hls, a = _remove_alpha(hls)
    rgb = colorsys.hls_to_rgb(*hls)
    return _add_alpha(rgb, a)


@_srgb_parser(multi=True, squeeze=True)
def rgb_to_hsv(*rgb_colors):
    """Converts a RGB-style color to a HSV color."""
    return [colorsys.rgb_to_hsv(*rgb) for rgb in rgb_colors]


def hsv_to_rgb(hsv):
    """Converts a HSV color to a RGB color."""
    hsv, a = _remove_alpha(hsv)
    rgb = colorsys.hls_to_rgb(*hsv)
    return _add_alpha(rgb, a)
