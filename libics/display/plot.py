import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from libics.data import arraydata, seriesdata
from libics.display import plotcfg
from libics.util import misc


###############################################################################


def _plot_meta(mpl_ax, plot_dim, cfg, data):
    if isinstance(data, arraydata.ArrayData):
        mpl_ax.set_xlabel(data.scale.quantity[0].mathstr())
        if data.data.ndim == 2:
            mpl_ax.set_ylabel(data.scale.quantity[1].mathstr())
        elif data.data.ndim == 1:
            ylabel = None
            if cfg.point is not None:
                ylabel = data.scale.quantity[cfg.point.ypos.dim].mathstr()
            elif cfg.curve is not None:
                ylabel = data.scale.quantity[cfg.curve.ypos.dim].mathstr()
            elif cfg.surface is not None:
                ylabel = data.scale.quantity[cfg.surface.ypos.dim].mathstr()
            elif cfg.contour is not None:
                ylabel = data.scale.quantity[cfg.contour.ypos.dim].mathstr()
            if ylabel is not None:
                mpl_ax.set_ylabel(ylabel)
        if plot_dim == 3:
            zlabel = None
            if cfg.point is not None:
                zlabel = data.scale.quantity[cfg.point.zpos.dim].mathstr()
            elif cfg.surface is not None:
                zlabel = data.scale.quantity[cfg.surface.zpos.dim].mathstr()
            elif cfg.contour is not None:
                zlabel = data.scale.quantity[cfg.contour.zpos.dim].mathstr()
            if zlabel is not None:
                mpl_ax.set_zlabel(zlabel)
    elif isinstance(data, seriesdata.SeriesData):
        xlabel, ylabel, zlabel = 3 * [None]
        if cfg.point is not None:
            xlabel = data.quantity[cfg.point.xpos.dim].mathstr()
            ylabel = data.quantity[cfg.point.ypos.dim].mathstr()
            if plot_dim == 3:
                zlabel = data.quantity[cfg.point.zpos.dim].mathstr()
        elif cfg.curve is not None:
            xlabel = data.quantity[cfg.curve.xpos.dim].mathstr()
            ylabel = data.quantity[cfg.curve.ypos.dim].mathstr()
            if plot_dim == 3:
                zlabel = data.quantity[cfg.curve.zpos.dim].mathstr()
        elif cfg.matrix is not None:
            xlabel = data.quantity[cfg.matrix.xpos.dim].mathstr()
            ylabel = data.quantity[cfg.matrix.ypos.dim].mathstr()
            if plot_dim == 3:
                zlabel = data.quantity[cfg.matrix.zpos.dim].mathstr()
        elif cfg.contour is not None:
            xlabel = data.quantity[cfg.contour.xpos.dim].mathstr()
            ylabel = data.quantity[cfg.contour.ypos.dim].mathstr()
            if plot_dim == 3:
                zlabel = data.quantity[cfg.contour.zpos.dim].mathstr()
        elif cfg.surface is not None:
            xlabel = data.quantity[cfg.surface.xpos.dim].mathstr()
            ylabel = data.quantity[cfg.surface.ypos.dim].mathstr()
            if plot_dim == 3:
                zlabel = data.quantity[cfg.surface.zpos.dim].mathstr()
        if xlabel is not None:
            mpl_ax.set_xlabel(xlabel)
        if ylabel is not None:
            mpl_ax.set_ylabel(ylabel)
        if zlabel is not None:
            mpl_ax.set_ylabel(zlabel)


def _cv_layered_1d_array(data):
    """
    Transposes a 1D array with items of arbitrary dimension (length) into
    a list (not necessarily Python list) of 1D arrays, where each array
    corresponds to one item dimension.

    Notes
    -----
    The following transposition is performed: [x, c] -> [c, x].
    If the original array items are scalar: [x] -> [c=1, x].
    """
    data_dim = len(data[0]) if hasattr(data[0], "__len__") else None
    if data_dim is None:
        data = [data]
    else:
        try:
            _data = np.array(data)
            if _data.dtype == "O":
                raise ValueError
            else:
                data = data.transpose((1, 0))
        except ValueError:
            _data = [np.empty(len(data), type(item))
                     for item in data[0]]
            for x in range(len(data)):
                for i in range(len(data[x])):
                    _data[i][x] = data[x][i]
            data = _data
    return data


def _cv_layered_2d_array(data):
    """
    Transposes a 2D array with items of arbitrary dimension (length) into
    a list (not necessarily Python list) of 2D arrays, where each array
    corresponds to one item dimension.

    Notes
    -----
    The following transposition is performed: [x, y, c] -> [c, x, y].
    If the original array items are scalar: [x, y] -> [c=1, x, y].
    """
    data_dim = len(data[0][0]) if hasattr(data[0][0], "__len__") else None
    if data_dim is None:
        data = [data]
    else:
        try:
            _data = np.array(data)
            if _data.dtype == "O":
                raise ValueError
            else:
                data = data.transpose((2, 0, 1))
        except ValueError:
            _data = [np.empty((len(data), len(data[0])), type(item))
                     for item in data[0][0]]
            for x in range(len(data)):
                for y in range(len(data[x])):
                    for i in range(len(data[x][y])):
                        _data[i][x][y] = data[x][y][i]
            data = _data
    return data


def _param_size(size, data):
    """
    Parameters
    ----------
    size : AttrSize
        Size attribute configuration.
    data : numpy.ndarray(2)
        2D data array.

    Returns
    -------
    xsize, ysize, zsize
        Matplotlib parameter: s.
    """
    s = {"x": None, "y": None, "z": None}
    for p in ("x", "y", "z"):
        if getattr(size, p + "dim") is not None:
            if getattr(size, p + "scale") == "const":
                s[p] = getattr(size, p + "map")
            elif size.xscale == "lin":
                s[p] = (getattr(size, p + "map")
                        * data[getattr(size, p + "dim")].flatten())
    return s["x"], s["y"], s["z"]


def _param_color(color, data):
    """
    Parameters
    ----------
    color : AttrColor
        Color attribute configuration.
    data : numpy.ndarray(2)
        2D data array.

    Returns
    -------
    data : numpy.ndarray
        Array values in color dimension.
    static_color, color_map, val_min, val_max, alpha
        Matplotlib parameters: c, cmap, vmin, vmax, alpha
    """
    data = data[color.dim]
    static_color, color_map = None, None
    if color.scale == "const":
        static_color = color.map
    elif color.scale == "lin":
        color_map = color.map
    val_min, val_max = color.min, color.max
    alpha = color.alpha
    return data, static_color, color_map, val_min, val_max, alpha


###############################################################################


def _plot_data_array_1d(mpl_ax, plot_dim, cfg, x, data):
    """
    Plot 1D array.

    Parameters
    ----------
    mpl_ax : matplotlib.axes.Axes
        Matplotlib axes to which to plot.
    plot_dim : 2 or 3
        Plot dimension.
    cfg : PlotCfg
        Plot configuration.
    x : numpy.ndarray(1)
        1D xdim array.
    data : numpy.ndarray(1)
        1D plottable data. Each data element may be higher
        dimensional.

    Returns
    -------
    mpl_artist : matplotlib.artist.Artist
        Matplotlib plot object.
    """
    mpl_artist = None
    data = _cv_layered_1d_array(data)
    # 2D plots
    if plot_dim == 2:
        # 2D point scatter plot
        if cfg.point is not None:
            if cfg.point.ypos is None:
                return
            yy = data[cfg.point.ypos.dim]
            s, c, marker, cmap, vmin, vmax, alpha = 7 * [None]
            if cfg.point.size is not None:
                for size in _param_size(cfg.point.size, data):
                    if size is not None:
                        s = size
                        break
            if cfg.point.color is not None and cfg.point.color.dim is not None:
                if cfg.point.color.scale == "const":
                    c = cfg.point.color.map
                elif cfg.point.color.scale == "lin":
                    cmap = cfg.point.color.map
                    c = data[cfg.point.color.dim]
                    vmin, vmax = cfg.point.color.min, cfg.point.color.max
                alpha = cfg.point.color.alpha
            if cfg.point.shape is not None:
                marker = cfg.point.shape
            mpl_artist = mpl_ax.scatter(
                x, yy, s=s, c=c, marker=marker, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha,
            )
        # 2D curve line plot
        if cfg.curve is not None:
            if cfg.curve.ypos is None:
                return
            yy = data[cfg.curve.ypos.dim]
            c, alpha, linestyle, linewidth = 4 * [None]
            if (cfg.curve.color is not None
                    and cfg.curve.color.scale == "const"):
                c = cfg.curve.color.map
                alpha = cfg.curve.color.alpha
            if cfg.curve.line is not None:
                linestyle = cfg.curve.line.shape
                linewidth = cfg.curve.line.thickness
            mpl_artist = mpl_ax.plot(
                x, yy, color=c, alpha=alpha,
                linestyle=linestyle, linewidth=linewidth
            )[0]
    # 3D plots
    elif plot_dim == 3:
        # 3D point scatter plot
        if cfg.point is not None:
            yy, zz = None
            if cfg.point.ypos is not None and cfg.point.ypos.dim is not None:
                yy = data[cfg.point.ypos.dim]
            if cfg.point.zpos is not None and cfg.point.zpos.dim is not None:
                zz = data[cfg.point.zpos.dim]
            s, c, marker, cmap, vmin, vmax, alpha = 7 * [None]
            if cfg.point.size is not None:
                for size in _param_size(cfg.point.size, data):
                    if size is not None:
                        s = size
                        break
            if cfg.point.color is not None and cfg.point.color.dim is not None:
                if cfg.point.color.scale == "const":
                    c = cfg.point.color.map
                elif cfg.point.color.scale == "lin":
                    cmap = cfg.point.color.map
                    c = data[cfg.point.color.dim]
                    vmin, vmax = cfg.point.color.min, cfg.point.color.max
                alpha = cfg.point.color.alpha
            if cfg.point.shape is not None:
                marker = cfg.point.shape
            mpl_artist = mpl_ax.scatter(
                x, yy, zz, s=s, c=c, marker=marker, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha,
            )
        # 3D curve line plot
        if cfg.curve is not None:
            if cfg.curve.ypos is None or cfg.curve.zpos is None:
                return
            yy, zz = data[cfg.curve.ypos.dim], data[cfg.curve.zpos.dim]
            c, alpha, linestyle, linewidth = 4 * [None]
            if (cfg.curve.color is not None
                    and cfg.curve.color.scale == "const"):
                c = cfg.curve.color.map
                alpha = cfg.curve.color.alpha
            if cfg.curve.line is not None:
                linestyle = cfg.curve.line.shape
                linewidth = cfg.curve.line.thickness
            mpl_artist = mpl_ax.plot(
                x, yy, zz, color=c, alpha=alpha,
                linestyle=linestyle, linewidth=linewidth
            )[0]
    return mpl_artist


def _plot_data_array_2d(mpl_ax, plot_dim, cfg, x, y, data):
    """
    Plot 2D array.

    Parameters
    ----------
    mpl_ax : matplotlib.axes.Axes
        Matplotlib axes to which to plot.
    plot_dim : 2 or 3
        Plot dimension.
    cfg : PlotCfg
        Plot configuration.
    x : numpy.ndarray(1)
        1D xdim array.
    y : numpy.ndarray(1)
        1D ydim array.
    data : numpy.ndarray(2)
        2D plottable data. Each data element may be higher
        dimensional.

    Returns
    -------
    mpl_artist : matplotlib.artist.Artist
        Matplotlib plot object.
    """
    mpl_artist = None
    xgrid, ygrid = np.meshgrid(x, y)
    data = _cv_layered_2d_array(data)
    # 2D plots
    if plot_dim == 2:
        # 2D point scatter plot
        if cfg.point is not None:
            xx, yy = xgrid.flatten(), ygrid.flatten()
            s, c, marker, cmap, vmin, vmax, alpha = 7 * [None]
            if cfg.point.size is not None:
                for size in _param_size(cfg.point.size, data):
                    if size is not None:
                        s = size
                        break
            if cfg.point.color is not None and cfg.point.color.dim is not None:
                _color = _param_color(cfg.point.color, data)
                _, c, cmap, vmin, vmax, alpha = _color
            if cfg.point.shape is not None:
                marker = cfg.point.shape
            mpl_artist = mpl_ax.scatter(
                xx, yy, s=s, c=c, marker=marker, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha,
            )
        # 2D color matrix plot
        if cfg.matrix is not None:
            c, cmap, vmin, vmax, alpha, edgecolors = 6 * [None]
            if (cfg.matrix.color is not None
                    and cfg.matrix.color.dim is not None):
                _color = _param_color(cfg.matrix.color, data)
                c, _, cmap, vmin, vmax, alpha = _color
            else:
                c = np.full_like(xgrid, 0)
                alpha = 0
            if (cfg.matrix.meshcolor is not None
                    and cfg.matrix.meshcolor.scale == "const"):
                edgecolors = cfg.matrix.meshcolor.cmap   # only static colors
            mpl_artist = mpl_ax.pcolormesh(
                x, y, c, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha,
                edgecolors=edgecolors
            )
        # 2D contour plot
        if cfg.contour is not None:
            z, levels, alpha, cmap, vmin, vmax = 6 * [None]
            if (cfg.contour.color is not None
                    and cfg.contour.color.dim is not None):
                _color = _param_color(cfg.contour.color, data)
                z, _, cmap, vmin, vmax, alpha = _color
            else:
                z = np.full_like(xgrid, 0)
                alpha = 0
            levels = cfg.contour.levels
            mpl_artist = mpl_ax.contourf(
                x, y, z, levels=levels, alpha=alpha, cmap=cmap,
                vmin=vmin, vmax=vmax
            )
    # 3D plots
    elif plot_dim == 3:
        # 3D point scatter plot
        if cfg.point is not None:
            xx, yy = xgrid.flatten(), ygrid.flatten()
            zz = None
            if cfg.point.zpos is not None and cfg.point.zpos.dim is not None:
                zz = data[cfg.point.zpos.dim].flatten()
            s, c, marker, cmap, vmin, vmax, alpha = 7 * [None]
            if cfg.point.size is not None:
                for size in _param_size(cfg.point.size, data):
                    if size is not None:
                        s = size
                        break
            if cfg.point.color is not None and cfg.point.color.dim is not None:
                _color = _param_color(cfg.point.color, data)
                _, c, cmap, vmin, vmax, alpha = _color
            if cfg.point.shape is not None:
                marker = cfg.point.shape
            mpl_artist = mpl_ax.scatter(
                xx, yy, zz, s=s, c=c, marker=marker, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha,
            )
        # 3D surface plot
        if cfg.surface is not None:
            z = None
            if (cfg.surface.zpos is not None
                    and cfg.surface.zpos.dim is not None):
                z = data[cfg.surface.zpos.dim]
            else:
                return
            cdata, color, cmap, vmin, vmax, alpha = 6 * [None]
            mcolor = None
            # Surface plot
            if (cfg.surface.color is not None
                    and cfg.surface.color.dim is not None):
                _color = _param_color(cfg.surface.color, data)
                cdata, color, cmap, vmin, vmax, alpha = _color
                if (cfg.surface.meshcolor is not None
                        and cfg.surface.meshcolor.scale == "const"):
                    mcolor = cfg.surface.meshcolor.map
            # Wireframe plot
            else:
                if (cfg.surface.meshcolor is not None
                        and cfg.surface.meshcolor.dim is not None):
                    _color = _param_color(cfg.surface.meshcolor, data)
                    cdata, _, cmap, _, _, _ = _color
            # Surface plot
            if alpha != 0:
                mpl_artist = mpl_ax.plot_surface(
                    xgrid, ygrid, z, color=color, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolor=mcolor
                )
            # Wireframe plot
            else:
                cdata = (cdata - cdata.min()) / (cdata.max() - cdata.min())
                mpl_artist = mpl_ax.plot_surface(
                    xgrid, ygrid, z, facecolors=cdata
                )
                mpl_artist.set_facecolors((0, 0, 0, 0))
        # 3D contour plot
        if cfg.contour is not None:
            z, levels, alpha, cmap, vmin, vmax = 6 * [None]
            if (cfg.contour.color is not None
                    and cfg.contour.color.dim is not None):
                _color = _param_color(cfg.contour.color, data)
                z, _, cmap, vmin, vmax, alpha = _color
            else:
                z = np.full_like(x, 0)
                alpha = 0
            levels = cfg.contour.levels
            mpl_artist = mpl_ax.contour(
                xgrid, ygrid, z, levels=levels, alpha=alpha, cmap=cmap,
                vmin=vmin, vmax=vmax
            )
    return mpl_artist


def _plot_data_series(mpl_ax, plot_dim, cfg, data):
    """
    Plot series data (tabular).

    Parameters
    ----------
    mpl_ax : matplotlib.axes.Axes
        Matplotlib axes to which to plot.
    plot_dim : 2 or 3
        Plot dimension.
    cfg : PlotCfg
        Plot configuration.
    data : list or numpy.ndarray(1) or numpy.ndarray(2)
        1D plottable data. Each data element may be higher
        dimensional.

    Returns
    -------
    mpl_artist : matplotlib.artist.Artist
        Matplotlib plot object.
    """
    mpl_artist = None
    if plot_dim == 2:
        # 2D point scatter plot
        if cfg.point is not None:
            if cfg.point.xpos is None or cfg.point.ypos is None:
                return
            xx, yy = data[cfg.point.xpos.dim], data[cfg.point.ypos.dim]
            s, c, marker, cmap, vmin, vmax, alpha = 7 * [None]
            if cfg.point.size is not None:
                for size in _param_size(cfg.point.size, data):
                    if size is not None:
                        s = size
                        break
            if cfg.point.color is not None and cfg.point.color.dim is not None:
                if cfg.point.color.scale == "const":
                    c = cfg.point.color.map
                elif cfg.point.color.scale == "lin":
                    cmap = cfg.point.color.map
                    c = data[cfg.point.color.dim]
                    vmin, vmax = cfg.point.color.min, cfg.point.color.max
                alpha = cfg.point.color.alpha
            if cfg.point.shape is not None:
                marker = cfg.point.shape
            mpl_artist = mpl_ax.scatter(
                xx, yy, s=s, c=c, marker=marker, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha,
            )
        # 2D curve line plot
        if cfg.curve is not None:
            if cfg.curve.xpos is None or cfg.curve.ypos is None:
                return
            xx, yy = data[cfg.curve.xpos.dim], data[cfg.curve.ypos.dim]
            c, alpha, linestyle, linewidth = 4 * [None]
            if (cfg.curve.color is not None
                    and cfg.curve.color.scale == "const"):
                c = cfg.curve.color.map
                alpha = cfg.curve.color.alpha
            if cfg.curve.line is not None:
                linestyle = cfg.curve.line.shape
                linewidth = cfg.curve.line.thickness
            mpl_artist = mpl_ax.plot(
                xx, yy, color=c, alpha=alpha,
                linestyle=linestyle, linewidth=linewidth
            )[0]
        # 2D color matrix plot
        if cfg.matrix is not None:
            if cfg.matrix.xpos is None or cfg.matrix.ypos is None:
                return
            xx, yy = data[cfg.matrix.xpos.dim], data[cfg.matrix.ypos.dim]
            c, cmap, vmin, vmax, alpha = 5 * [None]
            if cfg.matrix.color is not None:
                c = data[cfg.matrix.color.dim]
                cmap = cfg.matrix.color.map
                vmin, vmax = cfg.matrix.color.min, cfg.matrix.color.max
                alpha = cfg.matrix.color.alpha
            else:
                c = np.full_like(xx, 0)
                alpha = 0
            mpl_artist = mpl_ax.tripcolor(
                xx, yy, c, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
            )
            if (cfg.matrix.meshcolor is not None
                    and cfg.matrix.meshcolor.scale == "const"):
                linestyle = cfg.matrix.mesh.shape
                linewidth = cfg.matrix.mesh.thickness
                mpl_artist = mpl_ax.triplot(
                    xx, yy, color=c, alpha=alpha,
                    linestyle=linestyle, linewidth=linewidth
                )
        # 2D contour plot
        if cfg.contour is not None:
            if cfg.contour.xdim is None or cfg.contour.ydim is None:
                return
            xx, yy = data[cfg.contour.xdim], data[cfg.contour.ydim]
            zz = data[cfg.contour.color.dim]
            cmap = cfg.contour.color.map
            vmin, vmax = cfg.contour.color.min, cfg.contour.color.max
            alpha = cfg.contour.color.alpha
            levels = cfg.contour.levels
            mpl_artist = mpl_ax.tricontour(
                xx, yy, zz, levels=levels, alpha=alpha, cmap=cmap,
                vmin=vmin, vmax=vmax
            )
    elif plot_dim == 3:
        # 3D point scatter plot
        if cfg.point is not None:
            if (cfg.point.xpos is None or cfg.point.ypos is None
                    or cfg.point.zpos is None):
                return
            xx, yy = data[cfg.point.xpos.dim], data[cfg.point.ypos.dim]
            zz = data[cfg.point.zpos.dim]
            s, c, marker, cmap, vmin, vmax, alpha = 7 * [None]
            if cfg.point.size is not None:
                for size in _param_size(cfg.point.size, data):
                    if size is not None:
                        s = size
                        break
            if cfg.point.color is not None and cfg.point.color.dim is not None:
                if cfg.point.color.scale == "const":
                    c = cfg.point.color.map
                elif cfg.point.color.scale == "lin":
                    cmap = cfg.point.color.map
                    c = data[cfg.point.color.dim]
                    vmin, vmax = cfg.point.color.min, cfg.point.color.max
                alpha = cfg.point.color.alpha
            if cfg.point.shape is not None:
                marker = cfg.point.shape
            mpl_artist = mpl_ax.scatter(
                xx, yy, zz, s=s, c=c, marker=marker, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha,
            )
        # 3D curve line plot
        if cfg.curve is not None:
            if (cfg.curve.xpos is None or cfg.curve.ypos is None
                    or cfg.curve.zpos is None):
                return
            xx, yy = data[cfg.curve.xpos.dim], data[cfg.curve.ypos.dim]
            zz = data[cfg.curve.zpos.dim]
            c, alpha, linestyle, linewidth = 4 * [None]
            if (cfg.curve.color is not None
                    and cfg.curve.color.scale == "const"):
                c = cfg.curve.color.map
                alpha = cfg.curve.color.alpha
            if cfg.curve.line is not None:
                linestyle = cfg.curve.line.shape
                linewidth = cfg.curve.line.thickness
            mpl_artist = mpl_ax.plot(
                xx, yy, zz, color=c, alpha=alpha,
                linestyle=linestyle, linewidth=linewidth
            )[0]
        # 3D surface plot
        if cfg.surface is not None:
            if (cfg.surface.xpos is None or cfg.surface.ypos is None
                    or cfg.surface.zpos is None):
                return
            xx, yy = data[cfg.surface.xpos.dim], data[cfg.surface.ypos.dim]
            zz = data[cfg.surface.zpos.dim]
            cdata, color, cmap, vmin, vmax, alpha = 6 * [None]
            mcolor = None
            # Surface plot
            if cfg.surface.color is not None:
                if cfg.surface.color.scale == "lin":
                    color = data[cfg.surface.color.dim]
                    cmap = cfg.surface.color.map
                else:
                    color = cfg.surface.color.map
                vmin, vmax = cfg.surface.color.min, cfg.surface.color.max
                alpha = cfg.surface.color.alpha
                if (cfg.surface.meshcolor is not None
                        and cfg.surface.meshcolor.scale == "const"):
                    mcolor = cfg.surface.meshcolor.map
            # Wireframe plot
            else:
                if (cfg.surface.meshcolor is not None
                        and cfg.surface.meshcolor.dim is not None):
                    cdata = data[cfg.surface.meshcolor.dim]
                    cmap = cfg.surface.meshcolor.map
            # Surface plot
            if alpha != 0:
                mpl_artist = mpl_ax.plot_trisurf(
                    xx, yy, zz, color=color, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolor=mcolor
                )
            # Wireframe plot
            else:
                cdata = (cdata - cdata.min()) / (cdata.max() - cdata.min())
                mpl_artist = mpl_ax.plot_trisurf(
                    xx, yy, zz, facecolors=cdata
                )
                mpl_artist.set_facecolors((0, 0, 0, 0))
        # 3D contour plot
        if cfg.contour is not None:
            if cfg.contour.xdim is None or cfg.contour.ydim is None:
                return
            xx, yy = data[cfg.contour.xdim], data[cfg.contour.ydim]
            z = data[cfg.contour.color.dim]
            cmap = cfg.contour.color.map
            vmin, vmax = cfg.contour.color.min, cfg.contour.color.max
            alpha = cfg.contour.color.alpha
            levels = cfg.contour.levels
            mpl_artist = mpl_ax.tricontour(
                xx, yy, z, levels=levels, alpha=alpha, cmap=cmap,
                vmin=vmin, vmax=vmax
            )
    return mpl_artist


###############################################################################


def plot(mpl_ax, plot_dim, cfg, data, _data_type_hint="series"):
    """
    Diverts function call depending on data type.

    Parameters
    ----------
    mpl_ax : matplotlib.axes.Axes
        Matplotlib plot axes.
    plot_dim : 2 or 3
        Plot dimension.
    cfg : PlotCfg
        Plot configuration
    data : tuple, list, numpy.ndarray, arraydata.ArrayData, \
           seriesdata.SeriesData
        Data to be plotted.
    _data_type_hint : "array2d", "array1d" or "series"
        Hint for ambiguous data types whether to interpret
        data as array or series, e.g. for 2D numpy arrays.

    Returns
    -------
    mpl_artist : matplotlib.artist.Artist
        Matplotlib plot object.
    """
    data_type = None
    x, y = None, None
    # Distinguish data types
    if isinstance(data, arraydata.ArrayData):
        _plot_meta(mpl_ax, plot_dim, cfg, data)
        x = np.linspace(data.scale.offset[0], data.scale.max[0],
                        num=data.data.shape[0], endpoint=False)
        if data.data.ndim == 1:
            data_type = "array1d"
        if data.data.ndim == 2:
            data_type = "array2d"
            y = np.linspace(data.scale.offset[1], data.scale.max[1],
                            num=data.data.shape[1], endpoint=False)
        data = data.data
    elif isinstance(data, seriesdata.SeriesData):
        data_type = "series"
        _plot_meta(mpl_ax, plot_dim, cfg, data)
    elif isinstance(data, list):
        data_type = "series"
    elif isinstance(data, np.ndarray):
        data_type = _data_type_hint
    # Call plot functions
    if data_type == "array2d":
        return _plot_data_array_2d(mpl_ax, plot_dim, cfg, x, y, data)
    elif data_type == "array1d":
        return _plot_data_array_1d(mpl_ax, plot_dim, cfg, x, data)
    elif data_type == "series":
        return _plot_data_series(mpl_ax, plot_dim, cfg, data)


###############################################################################


class Figure(object):

    """
    Parameters
    ----------
    figure_cfg : FigureCfg
        Figure configuration.
    plot_cfgs : list(PlotCfg)
        List of plot configurations.
    plot_style_cfg : PlotStyleCfg or None
        Matplotlib rc plot style configuration.
    data : list(obj)
        List of data to be plotted.
    """

    def __init__(self,
                 figure_cfg, plot_cfgs, plot_style_cfg=None, data=None):
        self.figure_cfg = misc.assume_construct_obj(
            figure_cfg, plotcfg.FigureCfg
        )
        self.plot_cfgs = misc.assume_list(plot_cfgs)
        self.plot_dim = {}
        self.plot_style_cfg = None
        self.mpl_fig = None
        self.mpl_gs = None
        self.mpl_ax_loc = []
        self.mpl_ax = {}
        self.mpl_art = {}
        self.data = misc.assume_list(data)

    def add_mpl_ax(self, loc, xspan=None, yspan=None, plot_dim=None):
        """
        Adds an internal reference to already created axes to avoid
        matplotlib axes-reuse deprecation warning.
        Also sets up matplotlib artist dictionary.

        Parameters
        ----------
        loc : tuple(int)
            Gridspec location.
        xspan, yspan : int
            Gridspec span (x, y).
        plot_dim: 2 or 3
            Plot dimension.
        """
        self.mpl_ax_loc.append(loc)
        if loc not in self.mpl_ax.keys():
            self.plot_dim[loc] = plot_dim
            projection = "3d" if plot_dim == 3 else None
            self.mpl_ax[loc] = self.mpl_fig.add_subplot(
                self.mpl_gs.new_subplotspec(loc, xspan, yspan),
                projection=projection
            )
            self.mpl_art[loc] = []
        elif plot_dim != self.plot_dim[loc]:
            raise ValueError("incompatible plot dimensions")

    def setup_mpl(self):
        """
        Sets up the matplotlib figure environment to enable plotting.
        """
        # Create matplotlib figure
        self.mpl_fig = plt.figure()
        # FIXME: why doesn't this work???
        # self.mpl_fig = plt.figure(
        #     figsize=self.figure_cfg.get_size(unit="in"),
        #     dpi=self.figure_cfg.get_resolution(unit="in")
        # )
        # Create matplotlib gridspec
        xmax, ymax = 0, 0
        for item in self.plot_cfgs:
            if np.isscalar(item.xgridspec):
                xmax = max(xmax, item.xgridspec + 1)
            else:
                xmax = max(xmax, *item.xgridspec)
            if np.isscalar(item.ygridspec):
                ymax = max(ymax, item.ygridspec + 1)
            else:
                ymax = max(ymax, *item.ygridspec)
        self.mpl_gs = mpl.gridspec.GridSpec(xmax, ymax, figure=self.mpl_fig)
        # Create matplotlib axes
        for i, plot_cfg in enumerate(self.plot_cfgs):
            loc, span = plot_cfg.get_gridspec_param()
            self.add_mpl_ax(
                loc, xspan=span[0], yspan=span[1],
                plot_dim=plot_cfg.get_plot_dim()
            )

    def plot(self, data=None):
        """
        Plots the figure.

        Parameters
        ----------
        data : list(obj) or None
            Data to be plotted. If None, uses the internal
            data variable.

        Raises
        ------
        AssertionError
            If data length is incompatible with plot
            configurations.
        """
        if data is not None:
            self.data = misc.assume_list(data)
        assert(len(self.data) == len(self.plot_cfgs))
        if self.mpl_fig is None:
            self.setup_mpl()
        for i, plot_cfg in enumerate(self.plot_cfgs):
            mpl_art = plot(
                self.mpl_ax[self.mpl_ax_loc[i]],
                self.plot_dim[self.mpl_ax_loc[i]],
                plot_cfg, self.data[i]
            )
            if mpl_art is not None and plot_cfg.label is not None:
                self.mpl_art[self.mpl_ax_loc[i]].append(
                    (mpl_art, plot_cfg.label)
                )

    def legend(self):
        """
        Creates legends as set by the `plot` method.
        """
        for loc, ax in self.mpl_ax.items():
            if not self.mpl_art[loc]:
                continue
            artists, labels = [], []
            for val in self.mpl_art[loc]:
                if (
                    isinstance(val, mpl.lines.Line2D)
                    or isinstance(val, mpl.collections.PathCollection)
                    or isinstance(val, mplot3d.art3d.Line3D)
                    or isinstance(val, mplot3d.art3d.Path3DCollection)
                ):
                    artists.append(val[0])
                    labels.append(val[1])
                # TODO: Implement colorbar creation and annotation
                elif len(self.mpl_art[loc]) == 1:
                    ax.set_title(val[1])
            if artists:
                ax.legend(artists, labels)


###############################################################################


_MplRcParam = None


def enable_plot_style(plot_style):
    """
    Sets the default matplotlib rc plot style.

    Parameters
    ----------
    plot_style : plotcfg.PlotStyleCfg
        Default plot style configuration.
    """
    global _MplRcParam
    assert(isinstance(plot_style, plotcfg.PlotStyleCfg))
    mpl_rc = plotcfg.cv_plotstylecfg_to_mplrc(plot_style)
    _MplRcParam = mpl.rc_context(rc=mpl_rc)


def disable_plot_style():
    """
    Reverts any matplotlib rc plot style changes made by calling
    `enable_plot_style`.
    """
    if _MplRcParam is not None:
        _MplRcParam.__exit__()
