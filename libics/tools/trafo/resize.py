# System Imports
import numpy as np
import scipy.ndimage


###############################################################################


def crop_image(mask):
    """
    Crops to the smallest rectangle containing all non-masked indices.
    Returns the slice for cropping.

    Parameters
    ----------
    mask : `np.ndarray(2, bool)`
        Mask where `True` corresponds to indices not to be cropped.
        TODO: generalize for multiple dimensions.

    Returns
    -------
    crop : `tuple((int, int))`
        Cropping index rectangle (min, max). Dimensions: `[ar_ndim, 2]`.

    Notes
    -----
    From `https://codereview.stackexchange.com/questions/132914`.
    Use :py:func:`libics.core.util.misc.cv_index_rect_to_slice()` to convert
    the index rectangle to slices.

    Examples
    --------
    >>> x = y = np.arange(7) - 3
    >>> X, Y = np.meshgrid(x, y, indexing="ij")
    >>> F = X**2 + Y**2
    >>> mask = F < 4
    >>> crop = crop_image(mask)
    >>> crop
    ((2, 5), (2, 5))
    >>> F[misc.cv_index_rect_to_slice(crop)]
    array([[2, 1, 2],
           [1, 0, 1],
           [2, 1, 2]], dtype=int32)
    """
    if mask.ndim != 2:
        raise ValueError("'mask' has to be two-dimensional")
    m, n = mask.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return ((row_start, row_end), (col_start, col_end))


def center_crop(center, crop_rect, shape=None):
    """
    Centers a crop index rectangle.
    """
    center = np.array(center)
    crop_rect = np.array(crop_rect).T
    crop_rect[-1] += 1
    half_size = np.max(np.abs(crop_rect.T - center), axis=0)
    lower, upper = center - half_size + 1, center + half_size
    lower[lower < 0] = 0
    if shape is not None:
        shape = np.array(shape)
        upper[upper > shape] = shape[upper > shape]
    return tuple((lower[i], upper[i]) for i in range(len(center)))


###############################################################################


def find_centroid(np_array):
    """
    Finds the centroid of a multi-dimensional numpy array.

    Parameters
    ----------
    np_array : numpy.ndarray
        Numpy array for which the centroid is calculated.

    Returns
    -------
    centroid_coord : tuple(int)
        Index coordinates of the centroid.
    """
    return scipy.ndimage.measurements.center_of_mass(np_array)


###############################################################################


def fit_to_aspect(np_array, aspect_ratio, crop, center,
                  fit_mode="enlarge", center_mode="center"):
    """
    Changes a crop position to follow an aspect ratio.

    Parameters
    ----------
    np_array : numpy.ndarray
        Image data.
    aspect_ratio : tuple(float)
        Cropped pixel ratios between image dimensions.
    crop : tuple(tuple(int))
        Target crop coordinates with format
        `((ind_min_x, ind_min_y), (ind_max_x, ind_max_y))`.
    center : tuple(int)
        Center index coordinates.
    fit_mode : "enlarge", "reduce"
        Whether the crop should be enlarged or reduced
        with respect to the target crop.
    center_mode : "center", "off"
        Whether the given `center` coordinates should be in
        the center of the fitted crop image.

    Returns
    -------
    crop : tuple(tuple(int))
        Fitted crop coordinates with format
        `((ind_min_x, ind_min_y), (ind_max_x, ind_max_y))`.
    """
    # Transform to numpy arrays
    crop = np.array(crop)
    center = np.array(center)
    aspect_ratio = np.array(aspect_ratio)
    # Aspect ratio: normalized to x value
    if fit_mode == "enlarge":
        aspect_ratio = aspect_ratio / np.min(aspect_ratio)
    elif fit_mode == "reduce":
        aspect_ratio = aspect_ratio / np.max(aspect_ratio)
    # Relative scale: crop ratio normalized by target aspect ratio
    rel_scale = np.array(crop, dtype="float64")
    rel_scale = rel_scale[1] - rel_scale[0]
    rel_scale = rel_scale / rel_scale[0] / aspect_ratio
    # Relative coordinates: relative to given or cropped center
    rel_dist = np.full_like(center, -1)
    if center_mode == "center":
        pass
    elif center_mode == "off":
        center = np.mean(crop, axis=0)
    for ind, center_ind in enumerate(center):
        rel_dist[ind] = np.max((center_ind - crop[0, ind],
                                crop[1, ind] - center_ind))
    assert(np.all(rel_dist >= 0))
    # Resize
    rel_dist = rel_dist / rel_scale
    for ind, center_ind in enumerate(center):
        crop[0, ind] = center_ind - rel_dist[ind]
        crop[1, ind] = center_ind + rel_dist[ind]
    # Verify size
    flag_bounds_exceeded = False
    for ind, val in enumerate(crop[0]):
        if val < 0:
            crop[0, ind] = 0
            flag_bounds_exceeded = True
    for ind, val in enumerate(crop[1]):
        if val >= np_array.shape[ind]:
            crop[1, ind] = np_array.shape[ind] - 1
            flag_bounds_exceeded = True
    if flag_bounds_exceeded:
        # Implement action if necessary
        pass
    return crop


###############################################################################


def resize_on_mass(np_array, center="auto", total_mass=0.9,
                   aspect_ratio="auto", aspect_mode="enlarge"):
    """
    Finds the crop coordinates of a given numpy array for which each axis'
    sum exceeds the given relative `total_mass` parameter.

    Parameters
    ----------
    np_array : numpy.ndarray
        Numpy array to be resized.
    center : tuple(int) or "auto"
        Center coordinates of image. If `"auto"`, the centroid
        is used.
    total_mass : float
        Relative mass to be present in the cropped array.
    aspect_ratio : tuple(float) or "auto"
        Aspect ratio (e.g. 16:9) of cropped image (aspect ratio
        is not guaranteed). If `"auto"`, no aspect ratio is set.
    aspect_mode : "enlarge" or "reduce"
        If `"enlarge"`, the aspect ratio resizing enlarges the
        total mass crop. If `"reduce"`, it reduces the total mass
        crop.

    Returns
    -------
    crop : tuple(tuple(int))
        Crop coordinates with format
        `((ind_min_x, ind_min_y), (ind_max_x, ind_max_y))`.
    """
    if center == "auto":
        center = find_centroid(np_array)
    total_mass *= np.mean(np_array)
    # Find axes mass
    axes_mean = []
    for ax in range(len(np_array.shape)):
        axis = list(range(len(np_array.shape)))
        axis.remove(ax)
        axes_mean.append(np.mean(np_array, axis=tuple(axis)))
    # Find crop coordinates based on axes mass
    crop = np.full((2, len(np_array.shape)), -1)
    for dim, ax_mean in enumerate(axes_mean):
        it = 0
        while True:
            crop[0, dim] = np.max((center[dim] - it, 0))
            crop[1, dim] = np.min((center[dim] + it + 1, len(ax_mean)))
            if crop[0, dim] == 0 and crop[1, dim] == len(ax_mean):
                break
            _sum = np.sum(ax_mean[crop[0, dim]:crop[1, dim]])
            if _sum >= total_mass * len(ax_mean):
                break
            it += 1
    # Set aspect ratio
    if aspect_ratio != "auto":
        crop = fit_to_aspect(
            np_array, aspect_ratio, crop, center,
            fit_mode=aspect_mode, center_mode="center"
        )
    return crop


def _has_min_mass(ar, mask, min_mass):
    ret = (min_mass is None)
    if not ret:
        ret = (min_mass < ar[mask].sum() / ar.sum())
    return ret


def _has_min_val(ar, mask, min_val, peak_val=None):
    ret = (min_val is None)
    if not ret:
        if peak_val is None:
            peak_val = ar.max()
        ret = np.all(ar[np.logical_not(mask)] < min_val * peak_val)
    return ret


def resize_on_filter_maximum(
    np_array, min_mass=None, min_val=None,
    aspect_ratio=None, zero=True, factor=1
):
    """
    Performs maximum filters to obtain the area around a peak.

    Parameters
    ----------
    np_array : np.ndarray
        Non-negative array to be resized.
    min_mass : float
        Required relative mass within resized area.
    min_val : float
        Required relative value within resized area.
    aspect_ratio : float or None
        Resizing aspect ratio (y_shape / x_shape).
        None keeps the aspect ratio of the array
    zero : bool
        Whether to shift the array minimum to zero.
    factor : int
        Initial base size factor used for filtering.

    Returns
    -------
    crop : tuple(tuple(int))
        Crop coordinates with format
        `((ind_min_x, ind_min_y), (ind_max_x, ind_max_y))`.
    """
    if min_mass is None and min_val is None:
        raise ValueError("no minimum conditions set")
    if (
        (min_mass is not None and min_mass <= 0)
        or (min_val is not None and min_val <= 0)
    ):
        raise ValueError("invalid minimum conditions")
    if aspect_ratio is None:
        aspect_ratio = np_array.shape[1] / np_array.shape[0]
    if zero is True:
        np_array = np_array - np_array.min()

    peak_index = np.unravel_index(np_array.argmax(), np_array.shape)
    peak_val = np_array[peak_index]
    base_size = np.array([1.0, aspect_ratio])
    if aspect_ratio < 1:
        base_size = np.array([1.0 / aspect_ratio, 1.0])

    mask = np.full_like(np_array, False, dtype=bool)
    mask[peak_index] = True
    while(
        not _has_min_mass(np_array, mask, min_mass)
        or not _has_min_val(np_array, mask, min_val, peak_val=peak_val)
    ):
        size = tuple((factor * base_size).astype(int))
        filtered_array = scipy.ndimage.maximum_filter(
            np_array, size=size, mode="constant", cval=0
        )
        mask = (filtered_array == peak_val)
        factor += 1

    x, y = np.where(mask)
    crop = ((min(x), min(y)), (max(x) + 1, max(y) + 1))
    return crop


def resize_on_condition(np_array, cond="cut_all", **kwargs):
    """
    Finds the rectangle that contains/avoids all occurences of a given value.

    Parameters
    ----------
    np_array : np.ndarray
        Array to be resized.
    cond : callable or str
        Callable:
            Condition function that returns False if the
            respective hypersurface should be cut.
            Call signature:
                cond(np.array(np_array.ndim - 1)) -> bool.
        "cut_all", "cut_any":
            Remove boundary hypersurfaces (only, partially)
            containing kwargs["val"].
        "keep_all", "keep_any":
            Remove boundary hypersurfaces not (only, partially)
            containing kwargs["val"].
    val
        See cond -> "cut", "keep".

    Returns
    -------
    crop : np.array(2, int)
        Crop coordinates with format
        `((ind_min_x, ind_min_y, ...), (ind_max_x, ind_max_y, ...))`.
    """
    if isinstance(cond, str) and "val" in kwargs:
        val = kwargs["val"]
        if cond == "cut_any":
            if np.isnan(val):
                def cond(x):
                    return not np.any(np.isnan(x))
            else:
                def cond(x):
                    return not (val in x)
        elif cond == "cut_all":
            if np.isnan(val):
                def cond(x):
                    return not np.all(np.isnan(x))
            else:
                def cond(x):
                    return not np.all(x == val)
        elif cond == "keep_any":
            if np.isnan(val):
                def cond(x):
                    return np.any(np.isnan(x))
            else:
                def cond(x):
                    return val in x
        elif cond == "keep_all":
            if np.isnan(val):
                def cond(x):
                    return np.all(np.isnan(x))
            else:
                def cond(x):
                    return np.all(x == val)
    ind_max = np.array(np_array.shape)
    ind_min = np.zeros(len(ind_max), dtype=int)
    for dim in range(len(ind_max)):
        for ind in range(ind_max[dim]):
            if cond(np.take(np_array, ind, axis=dim)):
                break
            else:
                ind_min[dim] += 1
    for dim in range(len(ind_max)):
        for ind in reversed(range(ind_min[dim], ind_max[dim])):
            if cond(np.take(np_array, ind, axis=dim)):
                break
            else:
                ind_max[dim] -= 1
    return np.array([ind_min, ind_max])


###############################################################################


if __name__ == "__main__":

    # Setup test data
    ar = np.array([[np.exp(-((x - 100)**2 + (y - 100)**2) / (2 * 40**2))
                    for x in range(200)]
                   for y in range(200)])
    total_masses = [0.3, 0.5, 0.5, 0.5, 0.5, 0.9]
    aspect_ratios = [(1, 1), (1, 1), (1, 1), (1, 2), "auto", (1, 1)]
    aspect_modes = ["enlarge", "enlarge", "reduce",
                    "enlarge", "enlarge", "enlarge"]

    # Find crops for different conditions
    crops = [resize_on_mass(ar, center="auto",
                            total_mass=total_masses[it],
                            aspect_ratio=aspect_ratios[it],
                            aspect_mode=aspect_modes[it])
             for it in range(len(total_masses))]

    # Plot crops
    from matplotlib import pyplot, patches
    fig, ax = pyplot.subplots(1)
    ax.imshow(ar)
    rects = [patches.Rectangle(
                crop[0], crop[1, 0] - crop[0, 0], crop[1, 1] - crop[0, 1],
                linewidth=5, facecolor="none", edgecolor="C{:d}".format(it),
                label="{:s}, {:s}, {:s}".format(
                    str(total_masses[it]), str(aspect_ratios[it]),
                    str(aspect_modes[it])
                ))
             for it, crop in enumerate(crops)]
    for rect in rects:
        ax.add_patch(rect)
    ax.legend()
    pyplot.show()
