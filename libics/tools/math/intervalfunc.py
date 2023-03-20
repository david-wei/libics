import collections
import numpy as np
import sys

from libics.core.data.arrays import ArrayData


###############################################################################
# API
###############################################################################


class IntervalSeries():

    """
    Container class for a series of interval functions.
    """

    KEYS = {"func", "t0", "dt", "y0", "y1", "args", "kwargs"}

    def __init__(self):
        self._data = collections.defaultdict(list)

    @property
    def size(self):
        return len(self._data["func"])

    def append(
        self, *data,
        func=None, t0=None, dt=None, y0=None, y1=None, args=None, kwargs=None
    ):
        """
        Appends interval function data.
        """
        # List of dicts
        if len(data) > 0:
            for row in data:
                for k in self.KEYS:
                    self._data[k].append(row[k] if k in row else None)
        # Single item
        elif t0 is None or np.isscalar(t0):
            for k in self.KEYS:
                self._data[k].append(locals()[k])
        # Dict of lists
        else:
            for k in self.KEYS:
                self._data[k].extend(locals()[k])

    def get_times(self, iv_gap=0):
        """
        Calculates the time intervals from specified start times and durations.
        """
        # Initialize
        t0s = self._data["t0"].copy()
        dts = self._data["dt"].copy()
        t1s = self.size * [None]
        # Reconstruct start and end times of each interval
        if t0s[0] is None:
            t0s[0] = 0
        for i in range(self.size):
            # Current t0 not set
            if t0s[i] is None:
                # Initialization implies i > 0
                t0s[i] = t1s[i-1] + iv_gap
            # Current t1 not set
            if t1s[i] is None:
                if dts[i] is not None:
                    # Prior code guarantees t0 is set
                    t1s[i] = t0s[i] + dts[i]
                elif i < self.size and t0s[i+1] is not None:
                    t1s[i] = t0s[i+1] - iv_gap
                else:
                    raise ValueError(f"`t1` cannot be set at index {i}")
        return t0s, t1s, dts

    def get_blocks(self, iv_gap=0):
        """
        Calculates the block intervals.
        """
        t0s = np.arange(self.size, dtype=float)
        t0s, t1s = t0s + iv_gap / 2, t0s + 1 - iv_gap / 2
        return t0s, t1s, self.get_times()[2]

    def get_data(self, mode="val", iv_gap=0, num=64):
        """
        Gets the evaluated data traces.

        Parameters
        ----------
        mode : `str`
            `"val"`: Independent variable is set by value.
            `"block"`: Independent variable is set by block.
        iv_gap : `float`
            Default gap between subsequent intervals.
            `val` mode: Used when start times are not explicitly specified.
        num : `int`
            Points per interval trace.

        Returns
        -------
        ivs : `list(ArrayData(1, float))`
            Traces of intervals with length `self.size`.
        gaps : `list(ArrayData(1, float))`
            Traces of gaps between intervals with length `self.size - 1`.
        """
        # Set mode
        if mode == "val" or mode == "value":
            t0s, t1s, dts = self.get_times(iv_gap=iv_gap)
            rescales = len(dts) * [True]
        elif mode == "block":
            t0s, t1s, dts = self.get_blocks(iv_gap=iv_gap)
            rescales = [1/x for x in dts]
        else:
            raise ValueError(f"invalid `mode` {mode}")
        # Initialize variables
        ivs = []
        gaps = []
        for i in range(self.size):
            # Parse parameters
            func = assume_func(self._data["func"][i])
            t0, t1 = t0s[i], t1s[i]
            y0, y1 = self._data["y0"][i], self._data["y1"][i]
            if y0 is None:
                y0 = self._data["y1"][i - 1]
            if y1 is None:
                y1 = y0
            args, kwargs = self._data["args"][i], self._data["kwargs"][i]
            if args is None:
                args = tuple()
            if kwargs is None:
                kwargs = dict()
            # Set data
            iv_tcont = np.linspace(t0, t1, num=num)
            iv_ad = ArrayData(func(
                iv_tcont, y0, y1, *args, t0=t0, t1=t1,
                rescale=rescales[i], **kwargs
            ))
            iv_ad.set_dim(0, points=iv_tcont)
            ivs.append(iv_ad)
            if i > 0:
                last_t1 = t1s[i-1]
                last_y1 = self._data["y1"][i-1]
                gap_ad = ArrayData(np.array([last_y1, y0]))
                gap_ad.set_dim(0, points=np.array([last_t1, t0]))
                gaps.append(gap_ad)
        return ivs, gaps


###############################################################################
# Base
###############################################################################


def interval_func(*rescale_param):
    """
    Interval function factory.

    Parameters
    ----------
    *rescale_param : `int` or `str`
        Function call parameters which should be rescaled by the interval
        length `dt`.
        Use `int` to specify the indices of positional arguments.
        Use `str` to specify the keys of keyword arguments.

    Returns
    -------
    _ivf_factory : `callable`
        Decorator for function `func`.
        Functions with call signature: `func(t, y0, y1, *args)`,
        where `t` is the independent variable defined on the interval `[0, 1]`,
        `y0` and `y1` are the functional values at `t = 0` and `t = 1`.

    Examples
    --------
    To specify positional argument `p1`:

    >>> @interval_func(1)
    >>> def func(t, y0, y1, p0, p1, p2, k0=0, k1=1, k2=2):
    ...     pass

    To specify keyword argument `"k2"`:

    >>> @interval_func("k2")
    >>> def func(t, y0, y1, p0, p1, p2, k0=0, k1=1, k2=2):
    ...     pass
    """
    rescale_args = tuple(filter(lambda x: isinstance(x, int), rescale_param))
    rescale_kwargs = tuple(filter(lambda x: isinstance(x, str), rescale_param))

    # Decorator
    def _ivf_factory(func):

        # Function
        def _ivf(
            t, y0, y1, *args, t0=0, t1=None, dt=1, rescale=True, **kwargs
        ):
            # Check end value
            if y1 is None:
                y1 = y0
            # Transform t onto [0, 1] interval
            if t1 is not None:
                dt = t1 - t0
            # Rescale parameters
            if rescale is True:
                rescale_factor = 1 / dt
            elif rescale is False:
                rescale_factor = 1
            else:
                rescale_factor = rescale
            if len(rescale_args) > 0:
                args = tuple(
                    x * rescale_factor if i in rescale_args else x
                    for i, x in enumerate(args)
                )
            if len(rescale_kwargs) > 0:
                kwargs = {
                    k: v * rescale_factor if k in rescale_kwargs else v
                    for k, v in kwargs.items()
                }
            t = (t - t0) / dt
            return func(t, y0, y1, *args, **kwargs)

        # Docstring
        s = (
            f"        Interval function for {func.__name__}\n"
            f"        \n"
            f"        Parameters\n"
            f"        ----------\n"
            f"        t : `Array` or `Scalar`\n"
            f"            Independent variable.\n"
            f"        y0, y1 : `float`\n"
            f"            Start, stop value.\n"
            f"        t0, t1, dt : `float`\n"
            f"            Interval start value/stop value/extent.\n"
            f"            Specifying `t1` takes precedence over `dt`.\n"
            f"        rescale : `bool` or `float`\n"
            f"            Whether to rescale scale-dependent function \n"
            f"            parameters.\n"
            f"            If `float`, uses given value for rescaling.\n"
        )
        if func.__doc__ is not None:
            s = s + func.__doc__
        _ivf.__doc__ = s
        return _ivf
    return _ivf_factory


def assume_func(func):
    """
    Returns an interval function within this module.

    Parameters
    ----------
    func : `str` or `callable`
        If `callable` returns input. If `str`, gets the module-scoped
        function by name.

    Returns
    -------
    func : `callable`
        Requested function.
    """
    if callable(func):
        return func
    else:
        return getattr(sys.modules[__name__], func)


###############################################################################
# Functions
###############################################################################

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Flat
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@interval_func()
def lin(t, y0, y1):
    a = y1 - y0
    c = y0
    return a * t + c


@interval_func(0)
def exp(t, y0, y1, tau):
    a = (y1 - y0) / (np.exp(1 / tau) - 1)
    c = y0 - a
    return a * np.exp(t / tau) + c


@interval_func(0)
def tanh(t, y0, y1, tau):
    c = (y0 + y1) / 2
    a = (y1 - c) / np.tanh(1 / tau)
    return a * np.tanh((2 * t - 1) / tau) + c


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Peaked
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@interval_func(1)
def gauss(t, y0, y1, yp, tau):
    if y1 != y0:
        raise ValueError("`y0` must be equal to `y1`")
    a = (yp - y0) / (1 - np.exp(-1 / 2 / tau**2))
    c = yp - a
    return a * np.exp(-(t - 1/2)**2 / 2 / (tau / 2)**2) + c


@interval_func(0)
def cosh(t, y0, y1, yp, tau):
    if y1 != y0:
        raise ValueError("`y0` must be equal to `y1`")
    a = (yp - y0) / (np.cosh(1 / tau) - 1)
    c = yp + a
    return -a * np.cosh((t - 1/2) / (tau / 2)) + c


@interval_func(1, 2, "tc0", "tc1")
def trapez(t, y0, y1, yp, tc0=0.3, tc1=None):
    if tc1 is None:
        tc1 = 1 - tc0
    if tc1 < tc0:
        raise ValueError("`tc0` must be smaller than `tc1`")
    a0 = (yp - y0) / tc0
    c0 = y0
    a1 = (y1 - yp) / (1 - tc1)
    c1 = y1 - a1
    return np.piecewise(
        t, [t < tc0, t > tc1],
        [lambda t: a0 * t + c0, lambda t: a1 * t + c1, lambda t: yp]
    )


@interval_func(1, 2, "tc0", "tc1")
def step(t, y0, y1, yp, tc0=0.3, tc1=None):
    if tc1 is None:
        tc1 = 1 - tc0
    if tc1 < tc0:
        raise ValueError("`tc0` must be smaller than `tc1`")
    return np.piecewise(
        t, [t < tc0, t > tc1],
        [lambda t: y0, lambda t: y1, lambda t: yp]
    )
