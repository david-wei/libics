import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import numpy as np
import os

from libics import env
from libics.core.data.arrays import ArrayData, SeriesData
from libics.core.data.types import Quantity, ValQuantity
from libics.core.util import misc
from libics.tools.plot import colors


###############################################################################
# Matplotlib configuration
###############################################################################


def use_style(name="libics"):
    """
    Sets a matplotlib plot style.

    Parameters
    ----------
    name : `str`
        Matplotlib style file name.
    """
    if name not in plt.style.available:
        name = os.path.join(
            env.DIR_MPL, misc.assume_endswith(name, ".mplstyle")
        )
        if not os.path.exists(name):
            raise FileNotFoundError(
                f"Matplotlib style file not found. "
                f"Put the .mplstyle file in the folder {env.DIR_MPL}"
            )
    plt.style.use(name)


###############################################################################
# Figure and axes
###############################################################################


def subplots(
    fig=None, figsize=None, axsize=None, axsize_offset=None, size_unit="in",
    nrows=1, ncols=1, **kwargs
):
    """
    Create a figure and a set of subplots.

    See matplotlib API:
    https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html.
    """
    # Parse parameters
    if size_unit == "mm":
        if figsize is not None:
            figsize = (figsize[0] / 25.4, figsize[1] / 25.4)
        if axsize is not None:
            axsize = (axsize[0] / 25.4, axsize[1] / 25.4)
        if axsize_offset is not None:
            axsize_offset = (axsize_offset[0] / 25.4, axsize_offset[1] / 25.4)
    # Make figure
    if fig is None:
        if figsize is None and axsize is not None:
            figsize = tuple(
                _n * _size + _offset
                for _n, _size, _offset
                in zip([ncols, nrows], axsize, axsize_offset)
            )
        fig = plt.figure(figsize=figsize)
    # Create axes
    axs = fig.subplots(nrows=nrows, ncols=ncols, **kwargs)
    return fig, axs


def style_figure(
    fig=None, ax_style=None,
    figsize=None, figsize_unit="in", dpi=None, tight_layout=None,
    edgecolor=None, facecolor=None, title=None
):
    """
    Styles figure and contained axes properties.

    See matplotlib API:
    https://matplotlib.org/api/_as_gen/matplotlib.figure.Figure.html.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`
        Matplotlib figure.
    ax_style : `dict`
        Keyword arguments for :py:func:`style_axes`.
        Is applied to all axes contained in the figure.
    figsize_unit : `str`
        `"in", "mm"`. Unit used to interpret the `figsize` argument.
    dpi : `float`
        Figure resolution in dots per inch (dpi).
    tight_layout : `bool` or `dict`
        See :py:meth:`matplotlib.figure.Figure.set_tight_layout`.
    edgecolor, facecolor : `color`
        Edge, face color of the figure rectangle.
    title : `str`
        Figure suptitle.
    """
    fig = plt.gcf() if fig is None else fig
    # Figure size
    if figsize is not None:
        if figsize_unit == "mm":
            figsize = (figsize[0] / 25.4, figsize[1] / 25.4)
        fig.set_size_inches(figsize)
    # Resolution
    if dpi is not None:
        fig.set_dpi(dpi)
    # Layout
    if tight_layout is True:
        fig.set_tight_layout(tight_layout)
    # Colors
    if edgecolor is not None:
        fig.set_edgecolor(edgecolor)
    if facecolor is not None:
        fig.set_facecolor(facecolor)
    # Title
    if title is not None:
        fig.suptitle(title)
    # Axes
    if ax_style is not None:
        for ax in fig.axes:
            style_axes(ax=ax, **ax_style)


def style_axes(
    ax=None,
    xmin=None, xmax=None, ymin=None, ymax=None,
    xlabel=None, ylabel=None, title=None,
    aspect=None, grid=None, legend=None,
    minorticks=None, **ticks
):
    """
    Styles axes properties.

    See matplotlib API: https://matplotlib.org/api/axes_api.html.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    xmin, xmax, ymin, ymax : `float`
        Axes limits.
    xlabel, ylabel, title : `str`
        Axes labels.
    aspect : `float`
        Axes data scale aspect ratio.
    grid : `bool` or `str`
        `True, False, "major", "minor", "both"`.
    legend : `bool`
        Parameter passed to legend call.
    minorticks : `bool`
        Whether to show minor ticks.
    **ticks
        See :py:func:`tick_params`.
    """
    ax = plt.gca() if ax is None else ax
    # Limits
    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(top=ymax, bottom=ymin)
    # Labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    # Aspect
    if aspect is not None:
        ax.set_aspect(aspect)
    # Grid
    if grid is not None:
        if isinstance(grid, bool):
            ax.grid(b=grid)
        else:
            ax.grid(which=grid)
    # Ticks
    if minorticks is not None:
        if minorticks is True:
            ax.minorticks_on()
        else:
            ax.minorticks_off()
    tick_params(ax=ax, axis="both", **ticks)
    # Legend
    if legend is not None and legend is not False:
        if legend is True:
            ax.legend()
        else:
            ax.legend(legend)


def tick_params(
    ax=None, axis="both",
    capstyle=None, **kwargs
):
    """
    Styles tick properties.

    See matplotlib API:
    https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.tick_params.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    axis : `str`
        `"both", "x", "y"`.
    capstyle : `str`
        `"projecting", "round", "butt"`.
    **kwargs
        See :py:meth:`matplotlib.axes.Axes.tick_params`.
    """
    ax = plt.gca() if ax is None else ax
    # Matplotlib API
    ax.tick_params(axis=axis, **kwargs)
    # Cap style
    if capstyle is not None:
        if not isinstance(capstyle, mpl.markers.CapStyle):
            capstyle = getattr(mpl.markers.CapStyle, capstyle)
        if axis in ["x", "both"]:
            for i in ax.xaxis.get_majorticklines():
                i._marker._capstyle = capstyle
            for i in ax.xaxis.get_minorticklines():
                i._marker._capstyle = capstyle
        if axis in ["y", "both"]:
            for i in ax.yaxis.get_majorticklines():
                i._marker._capstyle = capstyle
            for i in ax.yaxis.get_minorticklines():
                i._marker._capstyle = capstyle


def savefig(fp, *args, **kwargs):
    plt.savefig(fp, *args, **kwargs)
    return fp


def unsqueeze_axes(axs, ax_1d="col"):
    """
    Unsqueezes multiple matplotlib axes to an array[row, col].

    Parameters
    ----------
    axs : `Iter[matplotlib.axes.Axes]`
        Collection of matplotlib axes.
    ax_1d : `str`
        If `axs` is 1D, it is assumed that the array dimension is
        `"col"` or `"row"`.

    Returns
    -------
    axs : `np.ndarray(2, matplotlib.axes.Axes)`
        2D array of matplotlib axes. Dimensions: [row, col].
    """
    if not isinstance(axs, (np.ndarray, tuple, list)):
        axs = [[axs]]
    axs = np.array(axs)
    if axs.ndim == 1:
        if ax_1d == "col":
            axs = axs[np.newaxis, :]
        else:
            axs = axs[:, np.newaxis]
    elif axs.ndim != 2:
        raise ValueError("Invalid shape of `axs`")
    return axs


def remove_axes(*axs, enforce=False, on_empty=True):
    """
    (Conditionally) removes a matplotlib axes.

    Parameters
    ----------
    *axs : `matplotlib.axes.Axes`
        Matplotlib axes object.
    enforce : `bool`
        Whether to remove the axes unconditionally.
    on_empty : `bool`
        Whether to remove the axes if no artists are present.
    """
    for ax in axs:
        if isinstance(ax, mpl.axes.Axes):
            if enforce is True:
                ax.remove()
            elif on_empty and not ax.has_data():
                ax.remove()


###############################################################################
# Artists
###############################################################################


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 1D Plots
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def plot(
    *data, x=None, y=None, xerr=None, yerr=None,
    xnorm=None, ynorm=None, marker=None,
    xlabel=True, ylabel=True, label=None, title=None,
    ax=None, **kwargs
):
    """
    Generates a 1D plot.

    Supports scatter plots, line plots and 2D error bars.
    See matplotlib API:

    * https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot
    * https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.errorbar

    Parameters
    ----------
    *data : `array-like, ArrayData, SeriesData`
        Plots data depending on passed data.
        Supports var data of `ArrayData` or `SeriesData`.
    x, y, xerr, yerr : `array-like, ArrayData, SeriesData`
        Explicitly given plot data (or error) for each axis.
        Overwrites `*data`.
    xnorm, ynorm : `float, ValQuantity`
        Normalization value for plot data.
        If `Quantity`, sets an automatic label.
    marker : `str` or `object`
        Matplotlib markers.
        Using square brackets (`[<marker>]`) around the matplotlib marker
        creates a larger marker with a darker edge color.
        Shorthands: `"O" == "[o]"`.
    xlabel, ylabel, label, title : `str` or `bool`
        Labels for the various properties.
        If `True`, tries to automatically set a label.
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    **kwargs
        Keyword arguments passed to the plot function.
    """
    ax = plt.gca() if ax is None else ax
    # Interpret arguments
    xnorm, xlabel = _process_norm_param(xnorm, label=xlabel)
    ynorm, ylabel = _process_norm_param(ynorm, label=ylabel)
    p = _get_xy_from_data(*data, xlabel=xlabel, ylabel=ylabel)
    x, y, xlabel, ylabel = p["x"], p["y"], p["xlabel"], p["ylabel"]
    xerr, yerr = _get_array(xerr, yerr)
    # Process normalization
    if xnorm is not None:
        x = np.array(x) / xnorm
        if xerr is not None:
            xerr = np.array(xerr) / xnorm
    if ynorm is not None:
        y = np.array(y) / ynorm
        if yerr is not None:
            yerr = np.array(yerr) / ynorm
    # Process marker style
    if "color" not in kwargs or kwargs["color"] is None:
        kwargs["color"] = ax._get_lines.get_next_color()
    kwargs = _process_marker_param(marker, **kwargs)
    # Perform plot
    if xerr is None and yerr is None:
        art = ax.plot(x, y, label=label, **kwargs)
    # Perform errorbar plot
    else:
        art = ax.errorbar(x, y, xerr=xerr, yerr=yerr, label=label, **kwargs)
        _process_err_param(art, **kwargs)
    # Set labels
    if isinstance(xlabel, str):
        ax.set_xlabel(misc.capitalize_first_char(xlabel))
    if isinstance(ylabel, str):
        ax.set_ylabel(misc.capitalize_first_char(ylabel))
    if isinstance(title, str):
        ax.set_title(title)
    return art


def scatter(*args, marker="O", linestyle="None", **kwargs):
    """
    Wrapper for :py:func:`plot`.

    Defaults to markers instead of lines.
    """
    return plot(*args, linestyle=linestyle, marker=marker, **kwargs)


scatter.__doc__ += "\n\n" + plot.__doc__


def bar(
    *data, x=None, y=None, width=None, bar_orientation="vertical",
    xnorm=None, ynorm=None, linestyle="=",
    xlabel=True, ylabel=True, label=None, title=None,
    ax=None, **kwargs
):
    """
    Generates a 1D bar plot.

    See matplotlib API:

    * https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.bar

    Parameters
    ----------
    *data : `array-like, ArrayData, SeriesData`
        Plots data depending on passed data.
        Supports var data of `ArrayData` or `SeriesData`.
    x, y : `array-like, ArrayData, SeriesData`
        Explicitly given plot data for each axis.
        Overwrites `*data`.
    width : `array-like, float`
        Width of the bars in units of the `x` variable.
    bar_orientation : `str`
        Orientation of individual bars: `"vertical", "horizontal"`.
    xnorm, ynorm : `float, ValQuantity`
        Normalization value for plot data.
        If `Quantity`, sets an automatic label.
    linestyle : `str`
        Matplotlib linestyle.
    xlabel, ylabel, label, title : `str` or `bool`
        Labels for the various properties.
        If `True`, tries to automatically set a label.
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    **kwargs
        Keyword arguments passed to the plot function.
    """
    ax = plt.gca() if ax is None else ax
    # Interpret arguments
    xnorm, xlabel = _process_norm_param(xnorm, label=xlabel)
    ynorm, ylabel = _process_norm_param(ynorm, label=ylabel)
    p = _get_xy_from_data(*data, xlabel=xlabel, ylabel=ylabel)
    x, y, xlabel, ylabel = p["x"], p["y"], p["xlabel"], p["ylabel"]
    if width is None:
        bins = ArrayData(y)
        bins.set_dim(0, points=x)
        bins = bins.get_bins(0)
        width = bins[1:] - bins[:-1]
    elif np.isscalar(width):
        width = np.full_like(x, width)
    # Process normalization
    if xnorm is not None:
        x = np.array(x) / xnorm
        if width is not None:
            width = np.array(width) / xnorm
    if ynorm is not None:
        y = np.array(y) / ynorm
    # Process patch style
    if "color" not in kwargs or kwargs["color"] is None:
        kwargs["color"] = ax._get_lines.get_next_color()
    kwargs = _process_patch_param(linestyle, **kwargs)
    # Perform plot
    if bar_orientation == "horizontal":
        art = ax.barh(x, y, height=width, label=label, **kwargs)
        xlabel, ylabel = ylabel, xlabel
    else:
        art = ax.bar(x, y, width=width, label=label, **kwargs)
    # Set labels
    if isinstance(xlabel, str):
        ax.set_xlabel(misc.capitalize_first_char(xlabel))
    if isinstance(ylabel, str):
        ax.set_ylabel(misc.capitalize_first_char(ylabel))
    if isinstance(title, str):
        ax.set_title(title)
    return art


def barh(*args, bar_orientation="horizontal", **kwargs):
    """
    Wrapper for :py:func:`bar`.

    Defaults to horizontally instead of vertically oriented bars.
    Uses `plt.barh` instead of `plt.bar` nomenclature,
    e.g., inverted meaning of `(width, height)` or `(x..., y...)`.
    """
    __kwargs_param_reverse(kwargs, ["width", "height"])
    __kwargs_param_reverse(kwargs, ["xlabel", "ylabel"])
    __kwargs_param_reverse(kwargs, ["xnorm", "ynorm"])
    return bar(*args, bar_orientation=bar_orientation, **kwargs)


barh.__doc__ += "\n\n" + bar.__doc__


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2D Plots
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def pcolormesh(
    *data, x=None, y=None, c=None,
    xnorm=None, ynorm=None, cnorm=None,
    vmin=None, vmax=None, vdif=None, vcen=None,
    xlabel=True, ylabel=True, title=None,
    colorbar=None, cb_orientation="vertical", clabel=True, cticks=None,
    aspect=None, ax=None, **kwargs
):
    """
    Generates a 2D color plot.

    Uses the pcolormesh function.
    See matplotlib API:
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolormesh.html

    Parameters
    ----------
    *data : `array-like, ArrayData, SeriesData`
        Plots data depending on passed data.
        Supports var data of `ArrayData` or `SeriesData`.
    x, y : `array-like, ArrayData, SeriesData`
        Explicitly given var plot data for each axis.
        Overwrites `*data`.
    c : `array-like, ArrayData`
        Explicitly given color plot data.
        Overwrites `*data`.
    xnorm, ynorm, cnorm : `float, ValQuantity`
        Normalization value for plot data.
        If `Quantity`, sets an automatic label.
    vmin, vmax, vdif, vcen : `float`
        Defines the color map range.
        Minimum/maximum are given either directly (`vmin, vmax`)
        or are deduced from center and range (`vcen, vdif`).
        If `vdif is True`, the full range is automatically assigned.
    xlabel, ylabel, clabel, title : `str` or `bool`
        Labels for the various properties.
        If `True`, tries to automatically set a label.
    colorbar : `bool` or `matplotlib.axes.Axes`
        Flag whether to show color bar.
        If `bool`, specifies the parent axes.
        If `Axes`, specifies the color bar axes.
    cb_orientation : `str`
        `"horizontal", "vertical"`.
    cticks : `None` or `list(ticks)`
        Colorbar ticks.
    aspect : `float`
        Axes data scale aspect ratio.
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    **kwargs
        Keyword arguments passed to the plot function.
    """
    ax = plt.gca() if ax is None else ax
    # Interpret arguments
    xnorm, xlabel = _process_norm_param(xnorm, label=xlabel)
    ynorm, ylabel = _process_norm_param(ynorm, label=ylabel)
    cnorm, clabel = _process_norm_param(cnorm, label=clabel)
    p = _get_xyc_from_data(
        *data, xlabel=xlabel, ylabel=ylabel, clabel=clabel, use_bins=True
    )
    x, y, c = p["x"], p["y"], p["c"]
    xlabel, ylabel, clabel = p["xlabel"], p["ylabel"], p["clabel"]
    # Process normalization
    if xnorm is not None:
        x = np.array(x) / xnorm
    if ynorm is not None:
        y = np.array(y) / ynorm
    if cnorm is not None:
        c = np.array(c) / cnorm
    if vdif is not None:
        if vcen is None:
            vcen = 0
        if vdif is True:
            vdif = np.nanmax(np.abs(c - vcen))
        if vmin is None:
            vmin = vcen - vdif
        if vmax is None:
            vmax = vcen + vdif
    # Perform pcolormesh
    art = ax.pcolormesh(x, y, c, vmin=vmin, vmax=vmax, **kwargs)
    # Set labels
    if isinstance(xlabel, str):
        ax.set_xlabel(misc.capitalize_first_char(xlabel))
    if isinstance(ylabel, str):
        ax.set_ylabel(misc.capitalize_first_char(ylabel))
    if isinstance(title, str):
        ax.set_title(title)
    # Aspect
    if aspect is not None:
        ax.set_aspect(aspect)
    # Color bar
    if colorbar is not None and colorbar is not False:
        fig = ax.get_figure()
        if colorbar is True:
            cb = fig.colorbar(
                art, ax=ax, orientation=cb_orientation, ticks=cticks
            )
        else:
            cb = fig.colorbar(
                art, ax=ax, cax=colorbar, orientation=cb_orientation,
                ticks=cticks
            )
        if isinstance(clabel, str):
            cb.set_label(misc.capitalize_first_char(clabel))
    return art


def pcolorim(*args, aspect=1, **kwargs):
    """
    Wrapper for :py:func:`pcolormesh`.

    Defaults to a common aspect ratio.
    """
    return pcolormesh(*args, aspect=aspect, **kwargs)


pcolorim.__doc__ += "\n\n" + pcolormesh.__doc__


def contourf(
    *data, x=None, y=None, c=None,
    xnorm=None, ynorm=None, cnorm=None,
    vmin=None, vmax=None, vdif=None, vcen=None,
    xlabel=True, ylabel=True, title=None,
    colorbar=None, cb_orientation="vertical", clabel=True,
    aspect=None, ax=None, **kwargs
):
    """
    Generates a 2D color contour plot.

    Uses the contourf function.
    See matplotlib API:
    https://matplotlib.org/api/_as_gen/matplotlib.pyplot.contourf.html

    Parameters
    ----------
    *data : `array-like, ArrayData, SeriesData`
        Plots data depending on passed data.
        Supports var data of `ArrayData` or `SeriesData`.
    x, y : `array-like, ArrayData, SeriesData`
        Explicitly given var plot data for each axis.
        Overwrites `*data`.
    c : `array-like, ArrayData`
        Explicitly given color plot data.
        Overwrites `*data`.
    xnorm, ynorm, cnorm : `float, ValQuantity`
        Normalization value for plot data.
        If `Quantity`, sets an automatic label.
    vmin, vmax, vdif, vcen : `float`
        Defines the color map range.
        Minimum/maximum are given either directly (`vmin, vmax`)
        or are deduced from center and range (`vcen, vdif`).
        If `vdif is True`, the full range is automatically assigned.
    xlabel, ylabel, clabel, title : `str` or `bool`
        Labels for the various properties.
        If `True`, tries to automatically set a label.
    colorbar : `bool` or `matplotlib.axes.Axes`
        Flag whether to show color bar.
        If `bool`, specifies the parent axes.
        If `Axes`, specifies the color bar axes.
    cb_orientation : `str`
        `"horizontal", "vertical"`.
    aspect : `float`
        Axes data scale aspect ratio.
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    **kwargs
        Keyword arguments passed to the plot function.
    """
    ax = plt.gca() if ax is None else ax
    # Interpret arguments
    xnorm, xlabel = _process_norm_param(xnorm, label=xlabel)
    ynorm, ylabel = _process_norm_param(ynorm, label=ylabel)
    cnorm, clabel = _process_norm_param(cnorm, label=clabel)
    p = _get_xyc_from_data(
        *data, xlabel=xlabel, ylabel=ylabel, clabel=clabel, use_bins=False
    )
    x, y, c = p["x"], p["y"], p["c"]
    xlabel, ylabel, clabel = p["xlabel"], p["ylabel"], p["clabel"]
    # Process normalization
    if xnorm is not None:
        x = np.array(x) / xnorm
    if ynorm is not None:
        y = np.array(y) / ynorm
    if cnorm is not None:
        c = np.array(c) / cnorm
    if vdif is not None:
        if vcen is None:
            vcen = 0
        if vdif is True:
            vdif = np.nanmax(np.abs(c - vcen))
        if vmin is None:
            vmin = vcen - vdif
        if vmax is None:
            vmax = vcen + vdif
    # Perform contourf
    art = ax.contourf(x, y, c, vmin=vmin, vmax=vmax, **kwargs)
    # Set labels
    if isinstance(xlabel, str):
        ax.set_xlabel(misc.capitalize_first_char(xlabel))
    if isinstance(ylabel, str):
        ax.set_ylabel(misc.capitalize_first_char(ylabel))
    if isinstance(title, str):
        ax.set_title(title)
    # Aspect
    if aspect is not None:
        ax.set_aspect(aspect)
    # Color bar
    if colorbar is not None and colorbar is not False:
        fig = ax.get_figure()
        if colorbar is True:
            cb = fig.colorbar(art, ax=ax, orientation=cb_orientation)
        else:
            cb = fig.colorbar(
                art, ax=ax, cax=colorbar, orientation=cb_orientation
            )
        if isinstance(clabel, str):
            cb.set_label(misc.capitalize_first_char(clabel))
    return art


###############################################################################
# Patches
###############################################################################


def plot_rectangle(
    rect=None, center=None, size=None,
    linestyle="=", angle=None,
    ax=None, **kwargs
):
    """
    Plots a rectangle.

    The rectangle can be specified either by its left-bottom and right-top
    corners (`rect`) or by its `center` location and `size`.

    Parameters
    ----------
    rect : `[[float, float], [float], float]]`
        Rectangle corners, specified as `[[xmin, xmax], [ymin, ymax]]`.
        Takes precedence over `center` and `size`.
    center, size : `[float, float]`
        Rectangle center and size, specified as `[x, y]`.
    angle : `float`
        Rotation of rectangle.
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    **kwargs
        Keyword arguments passed to the plot function.
    """
    ax = plt.gca() if ax is None else ax
    # Interpret arguments
    if rect is not None:
        _xy = rect[0][0], rect[1][0]
        size = rect[0][1] - rect[0][0], rect[1][1] - rect[1][0]
    else:
        _xy = center[0] - size[0] / 2, center[1] - size[1] / 2
    # Process marker style
    if angle is not None:
        kwargs["angle"] = angle
    if "color" not in kwargs or kwargs["color"] is None:
        kwargs["color"] = ax._get_lines.get_next_color()
    kwargs = _process_patch_param(linestyle, **kwargs)
    # Plot patch
    patch = mpl.patches.Rectangle(_xy, *size, **kwargs)
    art = ax.add_patch(patch)
    return art


def plot_polygon(
    vertices=None, center=None, size=None, num=None,
    linestyle="=", angle=None,
    ax=None, **kwargs
):
    """
    Plots a regular or irregular polygon.

    An irregular polygon can be directly specified by its `vertices`.
    A regular polygon can be specified by its `center` location, `size`,
    and `num`ber of vertices.

    Parameters
    ----------
    vertices : `Iter[[float], float]]`
        Vertex coordinates of (irregular) polygon,
        specified as `[[x0, y0], [x1, y1], ...]`.
        Takes precedence over `center`, `size` and `num`.
    center : `[float, float]`
        (Regular) polygon center, specified as `[x, y]`.
    size : `float`
        Size (diameter) of polygon.
    num : `int`
        Number of vertices of (regular) polygon.
    angle : `float`
        Rotation of polygon (only applied to regular polygons).
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    **kwargs
        Keyword arguments passed to the plot function.
    """
    ax = plt.gca() if ax is None else ax
    # Process marker style
    if "orientation" not in kwargs and angle is not None:
        kwargs["orientation"] = angle
    if "color" not in kwargs or kwargs["color"] is None:
        kwargs["color"] = ax._get_lines.get_next_color()
    kwargs = _process_patch_param(linestyle, **kwargs)
    # Plot patch
    if vertices is None:
        patch = mpl.patches.RegularPolygon(
            center, num, radius=size/2, **kwargs
        )
    else:
        patch = mpl.patches.Polygon(vertices, **kwargs)
    art = ax.add_patch(patch)
    return art


def plot_ellipse(
    center=None, size=None,
    linestyle="=", angle=None,
    ax=None, **kwargs
):
    """
    Plots an ellipse.

    Parameters
    ----------
    center : `[float, float]`
        Center of ellipse, specified as `[x, y]`.
    size : `float` or `[float, float]`
        Diameter of ellipse in both dimensions.
        If `float`, plots a circle.
    angle : `float`
        Rotation of ellipse.
    ax : `matplotlib.axes.Axes`
        Matplotlib axes.
    **kwargs
        Keyword arguments passed to the plot function.
    """
    ax = plt.gca() if ax is None else ax
    # Interpret arguments
    if np.isscalar(size):
        size = [size, size]
    # Process marker style
    if angle is not None:
        kwargs["angle"] = angle
    if "color" not in kwargs or kwargs["color"] is None:
        kwargs["color"] = ax._get_lines.get_next_color()
    kwargs = _process_patch_param(linestyle, **kwargs)
    # Plot patch
    patch = mpl.patches.Ellipse(center, *size, **kwargs)
    art = ax.add_patch(patch)
    return art


###############################################################################
# Helper functions
###############################################################################


def _get_xy_from_data(*data, xlabel=True, ylabel=True):
    """
    Parameters
    ----------
    *data
        Input argument to 1D plot functions.
    xlabel, ylabel : `str` or `bool`
        If `True`, uses interpreted labels.
        If `False`, does not use labels.
        If `str`, uses string as label.

    Returns
    -------
    data_dict : `dict`
        Interpreted data with keys: `"x", "y", "xlabel", "ylabel"`.
    """
    data_dict = {
        "x": None, "y": None, "xlabel": xlabel, "ylabel": ylabel
    }
    # No interpretable data passed
    if len(data) == 0:
        pass
    # Single parameter
    elif len(data) == 1:
        # (AD)
        if isinstance(data[0], ArrayData):
            data_dict["x"] = data[0].get_points(0)
            data_dict["y"] = data[0].data
            data_dict["xlabel"] = data[0].var_quantity[0].labelstr(val=False)
            data_dict["ylabel"] = data[0].data_quantity.labelstr(val=False)
        # (SD)
        elif isinstance(data[0], SeriesData):
            data_dict["x"] = data[0].data[0]
            data_dict["y"] = data[0].data[1]
            data_dict["xlabel"] = data[0].quantity[0].labelstr(val=False)
            data_dict["ylabel"] = data[0].quantity[1].labelstr(val=False)
        # (AR)
        else:
            data_dict["x"] = np.arange(len(data[0]))
            data_dict["y"] = data[0]
    # Two parameters
    elif len(data) == 2:
        # (AD, ...)
        if isinstance(data[0], ArrayData):
            data_dict["x"] = data[0].data
            data_dict["xlabel"] = data[0].data_quantity.labelstr(val=False)
        # (SD, ...)
        elif isinstance(data[0], SeriesData):
            data_dict["x"] = data[0].data[0]
            data_dict["xlabel"] = data[0].quantity[0].labelstr(val=False)
        # (AR, ...):
        else:
            data_dict["x"] = data[0]
        # (..., AD)
        if isinstance(data[1], ArrayData):
            data_dict["y"] = data[1].data
            data_dict["ylabel"] = data[1].data_quantity.labelstr(val=False)
        # (..., SD)
        elif isinstance(data[1], SeriesData):
            data_dict["y"] = data[1].data[0]
            data_dict["ylabel"] = data[1].quantity[0]
        # (..., AR)
        else:
            data_dict["y"] = data[1]
    # Invalid arguments
    else:
        raise TypeError("too many arguments")
    if xlabel is not True:
        data_dict["xlabel"] = xlabel
    if ylabel is not True:
        data_dict["ylabel"] = ylabel
    return data_dict


def _get_xyc_from_data(
    *data, xlabel=True, ylabel=True, clabel=True, use_bins=True
):
    """
    Parameters
    ----------
    *data
        Input argument to 1D plot functions.
    xlabel, ylabel, clabel : `str` or `bool`
        If `True`, uses interpreted labels.
        If `False`, does not use labels.
        If `str`, uses string as label.
    use_bins : `bool` or `None`
        Whether to get bins instead of points.
        TODO: Only works for single-parameter `data`.

    Returns
    -------
    data_dict : `dict`
        Interpreted data with keys:
        `"x", "y", "c", "xlabel", "ylabel", "clabel"`.
    """
    data_dict = {
        "x": None, "y": None, "c": None,
        "xlabel": xlabel, "ylabel": ylabel, "clabel": clabel
    }
    # No interpretable data passed
    if len(data) == 0:
        pass
    # Single parameter
    elif len(data) == 1:
        # (AD)
        if isinstance(data[0], ArrayData):
            data_dict["x"], data_dict["y"] = (
                data[0].get_var_meshgrid_bins() if use_bins is True
                else data[0].get_var_meshgrid()
            )
            for k in ["x", "y"]:
                if np.issubdtype(data_dict[k].dtype, object):
                    try:
                        data_dict[k] = data_dict[k].astype(float)
                    except TypeError:
                        pass
            data_dict["c"] = data[0].data
            data_dict["xlabel"] = data[0].var_quantity[0].labelstr(val=False)
            data_dict["ylabel"] = data[0].var_quantity[1].labelstr(val=False)
            data_dict["clabel"] = data[0].data_quantity.labelstr(val=False)
        # (AR)
        else:
            _data0 = np.array(data[0])
            _x = (np.arange(-0.5, _data0.shape[0] + 0.5) if use_bins is True
                  else np.arange(_data0.shape[0]))
            _y = (np.arange(-0.5, _data0.shape[1] + 0.5) if use_bins is True
                  else np.arange(_data0.shape[1]))
            _meshgrid0 = np.meshgrid(_x, _y, indexing="ij")
            data_dict["x"], data_dict["y"] = _meshgrid0
            data_dict["c"] = _data0
    # Three parameters
    elif len(data) == 3:
        # (AD, ...)
        if isinstance(data[0], ArrayData):
            data_dict["x"] = data[0].data
            data_dict["xlabel"] = data[0].data_quantity.labelstr(val=False)
        # (SD, ...)
        elif isinstance(data[0], SeriesData):
            data_dict["x"] = data[0].data[0]
            data_dict["xlabel"] = data[0].quantity[0].labelstr(val=False)
        # (AR, ...):
        else:
            data_dict["x"] = np.array(data[0])
        # (..., AD, ...)
        if isinstance(data[1], ArrayData):
            data_dict["y"] = data[1].data
            data_dict["ylabel"] = data[1].data_quantity.labelstr(val=False)
        # (..., SD, ...)
        elif isinstance(data[1], SeriesData):
            data_dict["y"] = data[1].data[0]
            data_dict["ylabel"] = data[1].quantity[0].labelstr(val=False)
        # (..., AR, ...):
        else:
            data_dict["y"] = np.array(data[1])
        # (..., AD)
        if isinstance(data[2], ArrayData):
            data_dict["c"] = data[2].data
            data_dict["clabel"] = data[2].data_quantity.labelstr(val=False)
        # (..., AR)
        else:
            data_dict["c"] = np.array(data[2])
    # Invalid arguments
    else:
        raise TypeError("wrong number of arguments ({:d})".format(len(data)))
    if xlabel is not True:
        data_dict["xlabel"] = xlabel
    if ylabel is not True:
        data_dict["ylabel"] = ylabel
    if clabel is not True:
        data_dict["clabel"] = clabel
    return data_dict


def _get_array(*data, nd=None):
    """
    Parameters
    ----------
    *data : `ArrayData, SeriesData, np.ndarray, list, tuple, None`
        Array data in arbitrary format.
    nd : `int`
        Number of dimensions to pick if `SeriesData`.
        If `None`, retrieves 1D array. Otherwise shape is `(nd, shape[1])`.

    Returns
    -------
    *data : `np.ndarray, list, tuple, None`
        If a transform was necessary, returns a `np.ndarray`.
        Otherwise returns input.
    """
    if len(data) == 1:
        data = data
        if isinstance(data[0], ArrayData):
            return data[0].data
        elif isinstance(data[0], SeriesData):
            if nd is None:
                return data[0].data[0]
            else:
                return data[0].data[:nd]
        else:
            return data[0]
    else:
        return [_get_array(_d, nd=nd) for _d in data]


def _process_norm_param(
    norm, label=True
):
    if norm is not None:
        if label is True:
            if isinstance(norm, Quantity):
                label = norm.labelstr(val=False)
        if isinstance(norm, ValQuantity):
            norm = norm.val
    return norm, label


def _process_marker_param(
    marker, **kwargs
):
    """
    Processes custom marker options.

    Parameters
    ----------
    marker : `str`
        Matplotlib marker options.
    **kwargs
        Other matplotlib style options for plotting.

    Returns
    -------
    kwargs : `dict`
        Updated matplotlib style options.
    """
    marker_map = {
        "O": "[o]"
    }
    marker = marker_map.get(marker, marker)
    try:
        marker = misc.extract(marker, r"^\[(.*)\]$")
        has_edge = True
    except (TypeError, KeyError):
        has_edge = False
    if has_edge:
        if "markersize" not in kwargs:
            kwargs["markersize"] = mpl.rcParams["lines.markersize"] * 1.3
        if "markeredgewidth" not in kwargs:
            kwargs["markeredgewidth"] = kwargs["markersize"] / 7
        if "markeredgecolor" not in kwargs:
            if "markerfacecolor" not in kwargs:
                kwargs["markeredgecolor"] = kwargs["color"]
                kwargs["markerfacecolor"] = colors.rgb_whiten(
                    kwargs["markeredgecolor"]
                )
            else:
                if __kwargs_param_is_not_empty(kwargs, "markerfacecolor"):
                    kwargs["markeredgecolor"] = colors.rgb_blacken(
                        kwargs["markerfacecolor"]
                    )
                else:
                    kwargs["markeredgecolor"] = mplc.colorConverter.to_rgba(
                        kwargs["color"]
                    )
        else:
            if "markerfacecolor" not in kwargs:
                if __kwargs_param_is_not_empty(kwargs, "markeredgecolor"):
                    kwargs["markerfacecolor"] = colors.rgb_whiten(
                        kwargs["markeredgecolor"]
                    )
                else:
                    kwargs["markerfacecolor"] = colors.rgb_whiten(
                        kwargs["color"]
                    )
    kwargs["marker"] = marker
    return kwargs


def _process_patch_param(linestyle, **kwargs):
    if "ec" in kwargs:
        kwargs["edgecolor"] = kwargs.pop("ec")
    if "fc" in kwargs:
        kwargs["facecolor"] = kwargs.pop("fc")
    if "lw" in kwargs:
        kwargs["linewidth"] = kwargs.pop("lw")
    _color = mplc.colorConverter.to_rgba(kwargs.pop("color"))
    if linestyle == "=":
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = mpl.rcParams["lines.linewidth"]
        if "edgecolor" not in kwargs:
            if "facecolor" not in kwargs:
                kwargs["edgecolor"] = _color
                kwargs["facecolor"] = colors.rgb_whiten(kwargs["edgecolor"])
            else:
                if __kwargs_param_is_not_empty(kwargs, "facecolor"):
                    _ec = colors.rgb_blacken(kwargs["facecolor"])
                else:
                    _ec = mpl.colorConverter.to_rgba(_color)
                kwargs["edgecolor"] = _ec
        else:
            if "facecolor" not in kwargs:
                if __kwargs_param_is_not_empty(kwargs, "edgecolor"):
                    _fc = colors.rgb_whiten(kwargs["edgecolor"])
                else:
                    _fc = mpl.colorConverter.to_rgba(_color)
                kwargs["facecolor"] = _fc
        kwargs["linestyle"] = "-"
    else:
        if "edgecolor" not in kwargs:
            kwargs["edgecolor"] = _color
        if "facecolor" not in kwargs:
            kwargs["facecolor"] = _color
        kwargs["linestyle"] = linestyle
    return kwargs


def _process_err_param(art, **kwargs):
    if "solid_capstyle" not in kwargs:
        kwargs["solid_capstyle"] = mpl.rcParams["lines.solid_capstyle"]
    _, caplines, barlinecols = art
    for c in caplines:
        c.set_solid_capstyle(kwargs["solid_capstyle"])
    for b in barlinecols:
        b.set_capstyle(kwargs["solid_capstyle"])


def __kwargs_param_is_not_empty(kwargs, param):
    """Returns whether `kwargs["param"]` contains reasonable data."""
    if param not in kwargs:
        return False
    v = kwargs[param]
    if v is None:
        return False
    elif isinstance(v, str):
        if v == "" or v.lower() == "none":
            return False
    return True


class __UNAVAILABLE:
    pass


def __kwargs_param_reverse(kwargs, params):
    """
    Reverses the order of key-value pairs (in place).

    Parameters
    ----------
    kwargs : `dict`
        Keyword argument dictionary.
    params : `Iter[str]`
        Keyword argument names.
        If items in `params` are not present in `kwargs`, they are ignored.

    Examples
    --------
    >>> d = {"a": 0, "b": 1, "c": 2, "d": 3}
    >>> __kwargs_param_reverse(d, ["a", "b", "d"])
    {'c': 2, 'd': 0, 'b': 1, 'a': 3}
    >>> __kwargs_param_reverse(d, ["a", "b", "x"])
    {'c': 2, 'd': 3, 'x': 0, 'b': 1}
    """
    values = [kwargs.pop(p, __UNAVAILABLE) for p in params]
    for p, v in zip(reversed(params), values):
        if v is not __UNAVAILABLE:
            kwargs[p] = v
    return kwargs
