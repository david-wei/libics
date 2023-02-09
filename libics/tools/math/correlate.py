import numpy as np
from itertools import permutations, chain
from math import factorial

from libics.env.logging import get_logger
from libics.core.data.arrays import ArrayData
from libics.core.util import misc

LOGGER = get_logger("libics.tools.math.correlate")


###############################################################################
# Helper functions
###############################################################################


def _slice_by_lengths(lengths, the_list):
    idx = 0
    for length in lengths:
        new = the_list[idx:idx + length]
        idx += length
        yield new


def _partition(number):
    return {
        (x,) + y for x in range(1, number)
        for y in _partition(number-x)
    } | {(number,)}


def _subgroups(my_list):
    partitions = _partition(len(my_list))
    permed = [
        set(permutations(each_partition, len(each_partition)))
        for each_partition in partitions
    ]
    for each_tuple in chain(*permed):
        yield list(_slice_by_lengths(each_tuple, my_list))


def _get_partition(my_list, num_groups):
    """Gets prefactor of permutation"""
    # FIXME: highly inefficient
    filtered = []
    for perm in permutations(my_list, len(my_list)):
        for sub_group_perm in _subgroups(list(perm)):
            if len(sub_group_perm) == num_groups:
                # within each partition
                sorted_list = [sorted(i) for i in sub_group_perm]
                # by number of elements and first element of each partition
                sorted_list = sorted(sorted_list, key=lambda t: (len(t), t[0]))
                if sorted_list not in filtered:
                    filtered.append(sorted_list)
    return filtered


###############################################################################
# Sample-averaged autocorrelation
###############################################################################


def autocorrelate_single_dist(
    data, vdists, connected=True, vdist_err_warn=True, vdist_err_val=np.nan
):
    """
    Calculates the sample-averaged autocorrelation for one distance.

    Parameters
    ----------
    data : `Array`
        Sample data with dimensions: `[n_samples, ...]`.
    vdists : `Array[2, int]` or `Array[1, int]`
        Vectorial index distance for which to evaluate correlator.
        If `vdists` is 2D, dimensions: `[n_order - 1, data_ndim]`.
        If `vdists` is 1D and data is 1D, dimensions: `[n_order - 1]`.
        If `vdists` is 1D and data is >1D, assumes `n_order == 2`.
    connected : `bool`
        Whether to calculate the connected correlator,
        i.e., to subtract lower order correlators.
    vdist_err_warn : `bool` or `Exception`
        Sets the behavior when an invalid value for `vdists` is found.
        If `False`, silently returns `vdist_err_val`.
        If `True`, additionally logs a warning message.
        If `Exception`, raises the given exception.
    vdist_err_val : `scalar`
        Return value on `vdists` error (if no exception is raised).

    Returns
    -------
    corr : `float` or `complex`
        Sample-averaged autocorrelation at the specified relative distances.
    """
    # Parse data
    data_ar = np.array(data)
    data_shape = np.array(data_ar.shape[1:])
    data_ndim = len(data_shape)
    n_samples = len(data_ar)
    # Parse vectorial distances
    vdists = np.array(vdists, dtype=int)
    if vdists.ndim == 1:
        if data_ndim == 1:
            vdists = vdists[:, np.newaxis]
        else:
            vdists = vdists[np.newaxis, :]
    if vdists.ndim != 2 or vdists.shape[1] != data_ndim:
        raise ValueError("invalid shape of `vdists`")
    vdists = np.concatenate([[np.zeros(data_ndim, dtype=int)], vdists])
    order_ndim = len(vdists)
    # Set up working memory for intermediate product array
    tmp_starts = np.max(-vdists, axis=0)
    tmp_stops = np.min(data_shape - vdists, axis=0)
    tmp_starts[tmp_starts < 0] = 0
    tmp_stops[tmp_stops > data_shape] = data_shape[tmp_stops > data_shape]
    tmp_sizes = tmp_stops - tmp_starts
    if np.any(tmp_sizes <= 0):
        if vdist_err_warn is False:
            return vdist_err_val
        err_msg = "invalid `vdists`"
        if vdist_err_warn is True:
            LOGGER.warning(err_msg)
            return vdist_err_val
        else:
            raise vdist_err_warn(err_msg)
    tmp_dtype = complex if data_ar.dtype == complex else float
    tmp_shape = (order_ndim, n_samples) + tuple(tmp_sizes)
    tmp_ar = np.full(tmp_shape, np.nan, dtype=tmp_dtype)

    # Calculate intermediate product array
    for order_dim in range(order_ndim):
        _roi = [
            slice(tmp_starts[data_dim] + vdists[order_dim, data_dim],
                  tmp_stops[data_dim] + vdists[order_dim, data_dim])
            for data_dim in range(data_ndim)
        ]
        tmp_ar[order_dim] = data_ar[tuple([slice(None)] + _roi)]
    # Return bare (non-connected) autocorrelator
    if connected is False:
        corr = np.nanmean(np.prod(tmp_ar, axis=0))
    # Calculate connected autocorrelator
    else:
        order_dims = list(range(order_ndim))
        corr = 0
        for order_dim in order_dims:
            _coeff = (-1)**order_dim * factorial(order_dim)
            _groups = _get_partition(order_dims, order_dim + 1)
            for _group in _groups:
                _res = 1
                for _sub in _group:
                    _res = _res * np.nanmean(
                        np.prod(tmp_ar[_sub], axis=0), axis=0
                    )
                corr = corr + _coeff * _res
        corr = np.nanmean(corr)
    return corr


def autocorrelate(
    data, connected=True, n_order=2, max_vdist=None, center_val=None,
    vdist_err_val=np.nan, print_progress=False
):
    """
    Gets sample-averaged n-th order autocorrelation at multiple distances.

    Parameters
    ----------
    data : `Array`
        Sample data with dimensions: `[n_samples, ...]`.
    connected : `boolean`
        Whether to calculate the connected correlator,
        i.e., to subtract lower order correlators.
        Example for `n_order=2`:
        If `True`: <XY>-<X><Y>, if `False`: <XY>.
    n_order : `int`
        Order of correlator.
    max_vdist : `Array[int or None]` or `int` or `None`
        Maximum correlator distance for each data dimension.
        If scalar, uses the scalar for all dimensions.
        If `None`, uses the maximally possible distance
        for the respective dimension.
    center_val : `scalar` or `None`
        If not `None`, sets the correlator at the origin to the given value.
    vdist_err_val : `scalar`
        Correlator value at invalid distance (combinations).

    Returns
    -------
    corr : `ArrayData(float or complex)`
        Sample-averaged autocorrelator.
        Number of dimensions: `(n_order - 1) * (data.ndim - 1)`.
        Dimensions: `[dx1, dy1, dz1, ..., dx2, dy2, dz2, ..., ...]`.

    Examples
    --------
    Example for 1D nearest-neighbor-correlated data:
    >>> ar = np.tile([-1, 1], 3)
    >>> data = [ar, -ar]
    >>> corr = autocorrelate(data)
    >>> corr.data
    array([-1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.])
    """
    # Check data format
    n_order = int(n_order)
    if n_order < 2:
        raise ValueError("Correlation order `n_order` must be >2")
    try:
        data_ar = np.array(data)
        if data_ar.dtype == object:
            raise ValueError()
    except ValueError:
        raise ValueError(
            "All samples in the parameter `data` must have the same shape"
        )
    data_shape = data_ar.shape[1:]
    data_ndim = len(data_shape)

    # Check for metadata
    is_ad = isinstance(data[0], ArrayData)
    if is_ad:
        ad = data[0]
        # Check whether ArrayData steps are equidistant
        try:
            for dim in range(ad.ndim):
                ad.get_step(dim, check_mode=RuntimeError)
        except RuntimeError:
            LOGGER.warning("`data` does not contain equidistant steps")

    # Parse maximum relative coordinates
    max_vdist = misc.assume_list(max_vdist)
    if len(max_vdist) < data_ndim:
        _factor = int(np.ceil(data_ndim / len(max_vdist)))
        max_vdist = _factor * max_vdist
    max_vdist = [
        data_shape[i] - 1 if max_vdist[i] is None else max_vdist[i]
        for i in range(data_ndim)
    ]

    # TODO: make use of point symmetry of correlators
    # Setup correlation data container
    corr_shape = tuple((n_order - 1) * [2*x + 1 for x in max_vdist])
    corr_dtype = complex if data_ar.dtype == complex else float
    corr = ArrayData(np.full(corr_shape, np.nan, dtype=corr_dtype))
    for dim in range(corr.ndim):
        corr.set_dim(dim, center=0, step=1)
    if is_ad:
        q = ad.data_quantity.copy()
        q.name += " correlation"
        corr.set_data_quantity(quantity=q)
        for dim in range(corr.ndim):
            data_dim = dim % data_ndim
            order_dim = dim // data_ndim + 1
            q = ad.var_quantity[data_dim]
            if n_order > 2:
                q.name += f" [{order_dim:d}]"
            corr.set_var_quantity(dim, quantity=q)
    # All vectorial index distances with dimensions: [{...}, vdist]
    all_vdists = np.moveaxis(corr.get_var_meshgrid(), 0, -1).astype(int)

    # Iterate vectorial distances
    _iter = np.ndindex(*corr_shape)
    if print_progress:
        _iter = misc.iter_progress(_iter)
    for idx in _iter:
        vdists = all_vdists[idx].reshape((n_order - 1, -1))
        _c = autocorrelate_single_dist(
            data_ar, vdists, connected=connected,
            vdist_err_warn=False, vdist_err_val=vdist_err_val
        )
        corr[idx] = _c
    # Apply correct variable steps
    if is_ad:
        for dim in range(corr.ndim):
            data_dim = dim % data_ndim
            corr.set_dim(
                dim, center=0, step=ad.get_step(data_dim, check_mode=False)
            )
    # Apply center value
    if center_val is not None:
        idx = tuple(np.array(corr.shape) // 2)
        corr[idx] = center_val
    return corr


def autocorrelate_bootstrap(
    data, func=None, bs_groups=None, bs_size=None, seed=None,
    print_progress=True, **kwargs
):
    """
    Performs bootstrapping for autocorrelator functions.

    Parameters
    ----------
    data : `Array`
        Sample data with dimensions: `[n_samples, ...]`.
    func : `callable`
        Autocorrelation function to call:
        :py:func:`autocorrelate_single_dist` or :py:func:`autocorrelate`.
        Call signature: `func(resampled_data, **kwargs)`.
    bs_groups, bs_size : `int` or `None`
        Bootstrap parameters for
        the number of resampled groups
        and the sample size of each group.
        If `None`, uses the size of the data.
    seed : `int` or `None`
        Random number generator seed.
    print_progress : `bool`
        Whether to print a progress bar.
    **kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    res : `dict(str->Any)`
        Results dictionary containing:
    data : `list(ArrayData)`
        List of bootstrap autocorrelation calculation results.
    sem : `ArrayData`
        Standard error of the mean of the autocorrelation.

    Examples
    --------
    Example for 1D nearest-neighbor-correlated data with only two samples:
    >>> ar = np.tile([-1, 1], 3)
    >>> data = [ar, -ar]
    >>> corr_bs = autocorrelate_bootstrap(
    ...     data, func=autocorrelate, bs_groups=10, seed=0
    ... )
    >>> corr_bs["sem"]
    array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    """
    # Parse parameters
    bs_groups = len(data) if bs_groups is None else int(bs_groups)
    bs_size = len(data) if bs_size is None else int(bs_size)
    if func not in [autocorrelate, autocorrelate_single_dist]:
        raise ValueError(
            "`func` must be either `autocorrelate` "
            "or `autocorrelate_single_dist`"
        )
    # Perform bootstrapping
    rng = np.random.RandomState(seed=seed)
    corrs = []
    _iter = range(bs_groups)
    if print_progress:
        _iter = misc.iter_progress(_iter)
    for _ in _iter:
        resampled_idxs = rng.choice(
            np.arange(len(data)), size=bs_size, replace=True
        )
        resampled_data = [data[i] for i in resampled_idxs]
        corr = func(resampled_data, **kwargs)
        corrs.append(corr)
    sem = np.std(corrs, axis=0)
    # Package results
    res = {"data": corrs, "sem": sem}
    return res
