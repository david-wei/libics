import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Divider, Size
import numpy as np

from libics.env import logging
from libics.core.util import misc


###############################################################################


def make_fixed_axes(fig, rect):
    """
    Creates a matplotlib axes at a fixed position.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        Figure to create the axes in.
    rect : `Iter[float]`
        Axes rectangle (left, bottom, width, height) in units
        relative to the figure size.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        Created axes.
    """
    mpl_divider = Divider(
        fig, tuple(rect),
        [Size.Scaled(0), Size.Scaled(1)], [Size.Scaled(0), Size.Scaled(1)]
    )
    ax = fig.add_axes(
        mpl_divider.get_position(),
        axes_locator=mpl_divider.new_locator(1, 1)
    )
    return ax


###############################################################################


class SubfigSize:

    """
    Helper class representing a scalar size.

    Is used to distinguish between a relative or absolute size.

    Parameters
    ----------
    size : `float`
        Size value.
    rel : `bool`
        Whether `size` is interpreted as relative quantity.
    """

    def __init__(self, size=1, rel=True):
        self.size = size
        self.rel = rel

    def is_relative(self):
        return self.rel

    def is_fixed(self):
        return not self.rel


class SubfigMargins:

    """
    Helper class for subfigure layout margins along one dimension.

    Implements the `__getitem__` method to index margins, where `[0]`
    indexes the lower (outer) margin and `[1:]` indexes the inner margins.

    Parameters
    ----------
    low, high : `float`
        Margins on the lower and upper bounds in mm.
    center : `float` or `Iter[float]`
        Margins between sublayouts in mm.

    Attributes
    ----------
    left, bottom, right top : `float`
        Aliases for `low` and `high`.
    outer, inner : `float`
        Used to overwrite all outer/inner margins.
    """

    def __init__(self, **kwargs):
        self.low = 0.0
        self._center = np.zeros(1, dtype=float)
        self.high = 0.0
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val):
        val = np.array(val)
        if val.ndim == 0:
            val = val[np.newaxis]
        elif val.ndim > 1:
            raise ValueError("invalid `center`")
        self._center = val

    def __getattr__(self, name):
        if name in {"left", "top"}:
            return self.low
        elif name in {"right", "bottom"}:
            return self.high
        elif name in {"outer"}:
            return np.mean((self.low, self.high))
        elif name in {"inner"}:
            return np.mean(self.center)
        elif name in {"hcenter", "vcenter"}:
            return self.center
        else:
            raise AttributeError(f"invalid attribute `{str(name)}`")

    def __setattr__(self, name, val):
        if name in {"left", "top"}:
            self.low = val
        elif name in {"right", "bottom"}:
            self.high = val
        elif name in {"outer"}:
            self.low = self.high = val
        elif name in {"inner", "hcenter", "vcenter"}:
            self.center = val
        else:
            object.__setattr__(self, name, val)

    def __len__(self):
        return len(self.center)

    def __getitem__(self, idx):
        if np.isscalar(idx):
            if idx == 0:
                return self.left
            else:
                return self.center[(idx - 1) % len(self.center)]
        elif isinstance(idx, slice):
            center = np.concatenate([[self.left], self.center])
            if idx.start is None:
                idx = slice(0, idx.stop)
            if idx.stop is None:
                idx = slice(idx.start, len(center))
            elif idx.stop >= len(center):
                center = np.concatenate(
                    [[center[0]]]
                    + np.ceil(idx.stop / len(center[1:])).astype(int)
                    * [list(center[1:])]
                )[:idx.stop]
            return center[idx]

    def __str__(self):
        s = [self.low] + list(self.center) + [self.high]
        return str(s)

    def __repr__(self):
        return f"<'{self.__class__.__name__}' at {hex(id(self))}> {str(self)}"

    def get_outer_margin_size(self):
        """Gets the sum of the outer margins."""
        return self.low + self.high

    def get_total_margin_size(self, num=None):
        """Gets the sum of all margins assuming `num` sublayouts."""
        total_margin_size = float(self.low + self.high)
        if num is None:
            total_margin_size += np.sum(self.center)
        else:
            total_margin_size += np.sum([self[i] for i in range(1, num)])
        return total_margin_size


###############################################################################


class SubfigLayout:

    """
    Matplotlib fixed layout class.

    Can be used to layout nested axes in a figure with exact positioning.

    Parameters
    ----------
    fig : `mpl.figure.Figure`
        Matplotlib figure.
    subfig_rect : `(float, float, float, float)`
        Subfigure rectangle.
    left, right, top, bottom : `float`
        Subfigure outside margins in mm.
    hcenter, vcenter : `float` or `Iter[float]`
        Horizontal and vertical margins between sublayouts in mm.
        If `float`, applies the same margin between all sublayouts.
    sublayouts : `Iter[2, (SubfigLayout or True)]`
        2D grid of sublayouts.
        Can be another `SubfigLayout` or a `mpl.axes.Axes` object
        (if `True`, creates an axes).
    sublayout_widths, sublayout_heights : `Iter[1, SubfigSize]`
        Widths and heights of the sublayouts.
    """

    LOGGER = logging.get_logger("libics.tools.plot.layout.SubfigLayout")

    def __init__(self, **kwargs):
        # Margins
        self.hmargins = SubfigMargins()
        self.vmargins = SubfigMargins()
        # Contents
        self.superlayout = None
        self._sublayouts = np.full((1, 1), True, dtype=object)
        self._sublayout_widths = np.full(
            1, SubfigSize(1, rel=True), dtype=object
        )
        self._sublayout_heights = np.full(
            1, SubfigSize(1, rel=True), dtype=object
        )
        # Subfig properties
        self.fig = None
        self._subfig_rect = None
        prekws = list(filter(lambda x: x in ["outer", "inner"], kwargs.keys()))
        for k in prekws:
            setattr(self, k, kwargs.pop(k))
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Warnings
        warnings = list(filter(
            lambda x: x not in kwargs, ["fig", "subfig_rect"]
        ))
        if len(warnings) > 0:
            s = ", ".join(warnings)
            self.LOGGER.warning(f"No `{s}` arguments specified")

    # ++++++++++++++++++++++++++
    # Properties
    # ++++++++++++++++++++++++++

    def __getattr__(self, name):
        if name in {"left", "right", "hcenter"}:
            return getattr(self.hmargins, name)
        elif name in {"bottom", "top", "vcenter"}:
            return getattr(self.vmargins, name)
        elif name in {"outer", "inner"}:
            return np.mean([
                getattr(self.hmargins, name), getattr(self.vmargins, name)
            ])
        else:
            raise AttributeError(f"invalid attribute `{str(name)}`")

    def __setattr__(self, name, val):
        if name in {"left", "right", "hcenter"}:
            setattr(self.hmargins, name, val)
        elif name in {"bottom", "top", "vcenter"}:
            setattr(self.vmargins, name, val)
        elif name in {"outer", "inner"}:
            setattr(self.hmargins, name, val)
            setattr(self.vmargins, name, val)
        else:
            object.__setattr__(self, name, val)

    @property
    def sublayouts(self):
        return self._sublayouts

    @sublayouts.setter
    def sublayouts(self, val):
        val = np.array(val, dtype=object)
        if val.ndim != 2:
            if val.ndim > 2:
                raise ValueError("invalid sublayouts ndim")
            val = val[(2 - val.ndim) * (np.newaxis,) + (Ellipsis,)]
        self._sublayouts = val
        # Adjust sublayout sizes
        _dsize = self._sublayouts.shape[0] - len(self.sublayout_heights)
        if _dsize > 0:
            self.sublayout_heights = np.concatenate([
                self.sublayout_heights, [SubfigSize() for _ in range(_dsize)]
            ])
        elif _dsize < 0:
            self.sublayout_heights = self.sublayout_heights[:_dsize]
        _dsize = self._sublayouts.shape[1] - len(self.sublayout_widths)
        if _dsize > 0:
            self.sublayout_widths = np.concatenate([
                self.sublayout_widths, [SubfigSize() for _ in range(_dsize)]
            ])
        elif _dsize < 0:
            self.sublayout_widths = self.sublayout_widths[:_dsize]

    @property
    def sublayout_widths(self):
        return self._sublayout_widths

    @sublayout_widths.setter
    def sublayout_widths(self, val):
        val = np.array([
            item if isinstance(item, SubfigSize) else SubfigSize(item)
            for item in misc.assume_iter(val)
        ])
        self._sublayout_widths = val
        # Adjust sublayouts
        _dsize = len(self._sublayout_widths) - self._sublayouts.shape[1]
        if _dsize != 0:
            if _dsize > 0:
                _new_sublayouts = np.full(
                    (len(self._sublayout_heights),
                     len(self._sublayout_widths)),
                    np.nan, dtype=object
                )
                _new_sublayouts[:, :-_dsize] = self._sublayouts
            else:
                _new_sublayouts = self._sublayouts[:, :_dsize]
            self._sublayouts = _new_sublayouts

    @property
    def sublayout_heights(self):
        return self._sublayout_heights

    @sublayout_heights.setter
    def sublayout_heights(self, val):
        val = np.array([
            item if isinstance(item, SubfigSize) else SubfigSize(item)
            for item in misc.assume_iter(val)
        ])
        self._sublayout_heights = val
        # Adjust sublayouts
        _dsize = len(self._sublayout_heights) - self._sublayouts.shape[0]
        if _dsize != 0:
            if _dsize > 0:
                _new_sublayouts = np.full(
                    (len(self._sublayout_heights),
                     len(self._sublayout_widths)),
                    np.nan, dtype=object
                )
                _new_sublayouts[:, :-_dsize] = self._sublayouts
            else:
                _new_sublayouts = self._sublayouts[:, :_dsize]
            self._sublayouts = _new_sublayouts

    @property
    def subfig_rect(self):
        if self._subfig_rect is None:
            raise ValueError("subfig_rect is not set")
        return self._subfig_rect

    @subfig_rect.setter
    def subfig_rect(self, val):
        if val is not None:
            val = np.array(val, dtype=float)
            if len(val) == 2:
                val = np.concatenate([[0, 0], val])
            if len(val) != 4:
                raise ValueError(
                    "invalid subfig_rect "
                    "(must be [hoffset, voffset, width, height)"
                )
        self._subfig_rect = val

    @property
    def shape(self):
        return self._sublayouts.shape

    @property
    def nrows(self):
        return self.shape[0]

    @property
    def ncols(self):
        return self.shape[1]

    @property
    def figsize(self):
        return self.fig.get_size_inches() * 25.4

    def normalize_rect(self, rect_mm):
        figwidth, figheight = self.figsize
        return np.array([
            rect_mm[0] / figwidth, rect_mm[1] / figheight,
            rect_mm[2] / figwidth, rect_mm[3] / figheight
        ])

    # ++++++++++++++++++++++++++
    # Getters
    # ++++++++++++++++++++++++++

    def __getitem__(self, idx):
        return self.sublayouts[idx]

    def iter_axs(self):
        """
        Gets an iterator over all sublayout axes.

        Creates the axes if not yet created.
        """
        for row, col in np.ndindex(*self.shape):
            sublayout = self.sublayouts[row, col]
            if sublayout is None:
                continue
            if sublayout is True:
                yield self.get_ax(row, col)
            elif isinstance(sublayout, mpl.axes.Axes):
                yield sublayout
            else:
                yield sublayout.iter_axs()

    def get_sublayout_size(self, subfig_size, row=None, col=None):
        """
        Calculates the sublayout size.

        Parameters
        ----------
        subfig_size : `(float, float)`
            Size of full subfigure.
        row, col : `int` or `None`
            Sublayout indices for which to calculate size.
            If `None`, returns sizes of all sublayouts.

        Returns
        -------
        widths, heights : `float` or `np.ndarray(1, float)`
            Width and height of selected sublayout(s).
        """
        total_sublayouts_size = (
            np.array(subfig_size)
            - np.array([self.hmargins.get_total_margin_size(num=self.ncols),
                        self.vmargins.get_total_margin_size(num=self.nrows)])
        )
        # Find individual sublayout sizes
        sublayout_widths, sublayout_heights = [], []
        rel_widths, rel_heights = [], []
        for x in self.sublayout_widths:
            if x.is_fixed():
                sublayout_widths.append(x.size)
            else:
                sublayout_widths.append(np.nan)
                rel_widths.append(x.size)
        for x in self.sublayout_heights:
            if x.is_fixed():
                sublayout_heights.append(x.size)
            else:
                sublayout_heights.append(np.nan)
                rel_heights.append(x.size)
        sublayout_widths = np.array(sublayout_widths)
        sublayout_heights = np.array(sublayout_heights)
        # Convert relative to absolute sizes
        total_rel_size = total_sublayouts_size - np.array([
            np.nansum(sublayout_widths), np.nansum(sublayout_heights)
        ])
        rel_widths, rel_heights = (
            np.array(rel_widths) / np.sum(rel_widths) * total_rel_size[0],
            np.array(rel_heights) / np.sum(rel_heights) * total_rel_size[1]
        )
        sublayout_widths[np.isnan(sublayout_widths)] = rel_widths
        sublayout_heights[np.isnan(sublayout_heights)] = rel_heights
        # Return selected sublayout size
        if row is not None:
            sublayout_heights = sublayout_heights[row]
        if col is not None:
            sublayout_widths = sublayout_widths[col]
        return sublayout_widths, sublayout_heights

    def get_sublayout_rect(self, row=None, col=None, subfig_rect=None):
        """
        Gets the position and size of a sublayout.

        Parameters
        ----------
        row, col : `int` or `slice` or `None`
            Row/column indices of the requested sublayout.
            If `int`, returns a single rect (along resp. dim.).
            If `slice`, returns a single rect (along resp. dim.) spanning
            the sliced range.
            If `None`, returns a list of rects.
        subfig_rect : `Iter[float]` or `None`
            Subfigure rectangle. If `None`, uses internal variable.

        Returns
        -------
        rects : `np.ndarray(float)`
            Rectangle parameters (left, bottom, width, height)
            of the requested sublayout in mm.
            May have more dimensions `[..., 4]` if multiple rows/columns
            are requested.
        """
        if subfig_rect is None:
            subfig_rect = self.subfig_rect
        subfig_offset = np.array(subfig_rect[:2])
        subfig_size = np.array(subfig_rect[2:])
        # Get all sublayout sizes
        widths, heights = self.get_sublayout_size(
            subfig_size, row=None, col=None
        )
        # Get all sublayout offsets
        hoffsets = np.array([
            np.sum(self.hmargins[:col + 1]) + np.sum(widths[:col])
            for col in range(len(widths))
        ]) + subfig_offset[0]
        voffsets = np.array([
            subfig_size[1]
            - np.sum(self.vmargins[:row + 1]) - np.sum(heights[:row + 1])
            for row in range(len(heights))
        ]) + subfig_offset[1]
        # Slice rect
        if row is not None:
            if np.isscalar(row):
                heights = heights[row:row + 1]
                voffsets = voffsets[row:row + 1]
            elif isinstance(row, slice):
                voffsets = voffsets[row.start:row.start + 1]
                if row.stop is None:
                    row_stop = len(voffsets)
                else:
                    row_stop = row.stop % (len(voffsets) + 1)
                vhigh = voffsets[row_stop - 1] + heights[row_stop - 1]
                heights = np.array([vhigh])
            else:
                raise ValueError("invalid `row`")
        if col is not None:
            if np.isscalar(col):
                widths = widths[col:col + 1]
                hoffsets = hoffsets[col:col + 1]
            elif isinstance(row, slice):
                hoffsets = hoffsets[col.start:col.start + 1]
                if col.stop is None:
                    col_stop = len(hoffsets)
                else:
                    col_stop = col.stop % (len(hoffsets) + 1)
                hhigh = hoffsets[col_stop - 1] + widths[col_stop - 1]
                widths = np.array([hhigh])
            else:
                raise ValueError("invalid `col`")
        # Return rect
        _shape = len(heights), len(voffsets)
        rects = np.zeros(_shape + (4,), dtype=float)
        for i, j in np.ndindex(_shape):
            rects[i, j] = (hoffsets[i], voffsets[j], widths[i], heights[j])
        if col is not None:
            rects = rects[:, 0]
        if row is not None:
            rects = rects[0]
        return rects

    # ++++++++++++++++++++++++++
    # Setters
    # ++++++++++++++++++++++++++

    def make_ax(self, row=None, col=None, subfig_rect=None):
        """
        Creates a matplotlib axes at the given sublayout position.

        Checks if the sublayout axes was already created.
        Creates only axes if the sublayout item is `True`.

        Parameters
        ----------
        row, col : `int` or `None`
            Sublayout indices to create an axes in.
            If `None`, creates an axes for each sublayout item.
        subfig_rect : `Iter[float]` or `None`
            Subfigure rectangle. If `None`, uses internal variable.

        Returns
        -------
        ax : `mpl.axes.Axes`
            Created matplotlib axes.
            Is vectorial if multiple rows/columns were selected.
        """
        # Return multiple axes
        if row is None and col is None:
            axs = np.full(self.shape, None, dtype=object)
            for i, j in np.ndindex(*self.shape):
                axs[i, j] = self.make_ax(subfig_rect=subfig_rect, row=i, col=j)
            return axs
        elif col is None:
            axs = np.full(self.ncols, None, dtype=object)
            for j in range(self.ncols):
                axs[j] = self.make_ax(subfig_rect=subfig_rect, row=row, col=j)
            return axs
        elif row is None:
            axs = np.full(self.nrows, None, dtype=object)
            for i in range(self.nrows):
                axs[i] = self.make_ax(subfig_rect=subfig_rect, row=i, col=col)
            return axs
        # Existing axis
        sublayout = self.sublayouts[row, col]
        if isinstance(sublayout, mpl.axes.Axes):
            return sublayout
        elif sublayout is None or sublayout is False:
            return None
        # Create axis if sublayout is True
        rect = self.get_sublayout_rect(
            row=row, col=col, subfig_rect=subfig_rect
        )
        ax = make_fixed_axes(self.fig, self.normalize_rect(rect))
        self.sublayouts[row, col] = ax
        return ax
