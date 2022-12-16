import numpy as np
from scipy.integrate import trapz, cumtrapz

from libics.core.data.arrays import ArrayData


###############################################################################


def integrate_array(ar, bounds=False, x=None, dx=None, x0=None, axis=-1):
    """
    Performs a definite or indefinite 1D integral of an array.

    Parameters
    ----------
    ar : `Array`
        Function represented by an array.
    bounds : `bool` or `tuple(float)`
        `False`: performs indefinite integral.
        `True`: performs definite integral over full array.
        `tuple(float)`: integration bounds `[xmin, xmax]`.
    x : `Array(1, float)`
        Coordinates to integrate along. Overwrites `dx`.
        If not given, deduces `x` from `ar`.
    dx : `float`
        Coordinate steps to integrate along.
    x0 : `None` or `float`
        Used only for indefinite integration, where
        :math:`F(x) = \\int_{x_0}^{x} f(x') dx'`.
        If `None`, sets the integration constant to zero.
    axis : `int`
        Array dimension to integrate along.

    Returns
    -------
    ar_int : `Array` or `Number`
        If indefinite integration, returns same type as `ar`.
        If definite integration, removes one dimension of `ar`.
    """
    # Parse data
    is_ad = isinstance(ar, ArrayData)
    ad = ArrayData(ar)
    axis = axis % ad.ndim
    if x is None:
        if dx is None:
            x = ad.get_points(axis)
        else:
            x = np.arange(ar.shape[axis]) * dx
    indefinite_integration = bounds is False
    # Indefinite integration
    if indefinite_integration:
        ar_int = cumtrapz(ad, x=x, axis=axis, initial=0)
        if x0 is not None:
            idx_x0 = ad.cv_quantity_to_index(x0, axis)
            ar_int -= (
                ar_int[axis * (slice(None),) + (idx_x0,)]
                [(...,) + (ar_int.ndim - axis) * (np.newaxis,)]
            )
    # Definite integration
    else:
        if bounds is not True:
            idx_bounds = (
                ad.cv_quantity_to_index(bounds[0], axis),
                ad.cv_quantity_to_index(bounds[1], axis) + 1
            )
            ad = ad[axis * (slice(None),) + (slice(*idx_bounds),)]
            x = ad.get_points(axis)
        ar_int = trapz(ad, x=x, axis=axis)
    # Package result
    if is_ad and not np.isscalar(ar_int):
        ad_int = ad.copy_var()
        if not indefinite_integration:
            ad_int.rmv_dim(axis)
        ad_int.data = ar_int
        ar_int = ad_int
    return ar_int


def differentiate_array(ar, x=None, dx=None, axis=-1, edge_order=2):
    """
    Differentiates an array.

    Parameters
    ----------
    ar : `Array`
        Function represented by an array.
    x : `Array(1, float)`
        Coordinates to differentiate along. Overwrites `dx`.
        If not given, deduces `x` from `ar`.
    dx : `float`
        Coordinate steps to differentiate along.
    axis : `int`
        Array dimension to differentiate along.
    edge_order : `1` or `2`
        Numerical differentiation using first/second-order-accurate
        differences at the boundaries.

    Returns
    -------
    ar_dif : `Array`
        Differentiated array of same type as `ar`.
    """
    # Parse data
    is_ad = isinstance(ar, ArrayData)
    ad = ArrayData(ar)
    axis = axis % ad.ndim
    if x is None:
        if dx is None:
            x = ad.get_points(axis)
        else:
            x = np.arange(ar.shape[axis]) * dx
    # Perform differentiation
    ar_dif = np.gradient(ad, x, axis=axis, edge_order=edge_order)
    # Package result
    if is_ad:
        ad_dif = ad.copy_var()
        ad_dif.data = ar_dif
        ar_dif = ad_dif
    return ar_dif
