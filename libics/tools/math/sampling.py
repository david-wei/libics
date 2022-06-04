import numpy as np
import scipy


###############################################################################


def get_nonlinear_support_points(
    start, stop, nr_points, func, offset=0.1, method="slope", **kwargs
):
    """
    Creates a set of support points that are distributed to maximize sampling
    around regions of interest.

    The overall idea is to distribute the sampling points to obtain an
    equidistant spacing by some critertion. Choosing "slope" as an example,
    the returned sampling points are weighted with the respective slope of the
    function.

    Parameters
    ----------
    start : `float`
        Startpoint of scan. Will always be in final set of support points.
    stop : `float`
        Last point of scan. Will always be in final set of support points.
    nr_points : `int`
        Number of points in final set of support points.
    func : `callable`
        A known a priori function that we want to sample.
    offset : `float`
        Level of offset added to any background region of un-interest
        (meaning if we filter for "curvature", any region without curvature is
        of un-interest).
        If set to 0, no sample points can be found in the regions without
        interest.
    method : `str`
        Possible strings are `["slope", "curvature", "max", "equidistant"]`.
        The method defines the region of interest where the sample points are
        generated.
    **kwargs
        Additional arguments to be passed on to `func`.

    Returns
    -------
    x : `np.ndarray(float)`
        Support points ranging from start to stop with special weight given
        according to `method`.
    """
    # initially create an equally spaced oversampled support
    x_init = np.linspace(start, stop, nr_points*1000)

    # function that we want to sample
    y_func = func(x_init, **kwargs)

    # defining the cumulative sum by the given "method"
    if method in ("slope", "curvature", "max", "equidistant"):
        if method == "slope":
            der = np.gradient(y_func)
            y = (np.abs(der/max(der))+offset)/(1+offset)
        elif method == "curvature":
            cur = np.gradient(np.gradient(y_func))
            y = (np.abs(cur/max(cur))+offset)/(1+offset)
        elif method == "max":
            ymax = y_func
            y = (np.abs(ymax/max(ymax))+offset)/(1+offset)
        elif method == "equidistant":
            x = np.linspace(start, stop, nr_points)
            return x

        # defining the cumulative sum of the given method (scaled to 1)
        csum = np.cumsum(y)/max(np.cumsum(y))

        # interpolation of cumulatice sum. Needed to derive support points
        int_percentile = scipy.interpolate.interp1d(csum, x_init)

        # create array with equidistant support points in cumulative space
        per = np.linspace(1/(nr_points-1), 1, nr_points-1)

        # transforming the support points in cumulative space to real space
        x = int_percentile(per)
        x = list(x)
        x = np.array([x_init[0]] + x)

        return x

    else:
        raise ValueError(
            f"`get_support_points` can not be operated as `method` "
            f"({str(method)}) is not defined. "
            f'The choices are "slope", "curvature" or "max".'
        )
