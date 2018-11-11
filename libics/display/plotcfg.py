import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from libics import cfg
from libics.data import arraydata, seriesdata
from libics.util import misc


###############################################################################


class AttrBase(object):

    """
    Base class for plot value representations. Stores the required attributes.
    """

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


class AttrPosition(AttrBase):

    """
    Position attribute.

    Parameters
    ----------
    dim : int
        Data dimension.
    scale : "lin", "log"
        Scale type.
    min, max : float
        Minimum, maximum plotted position.
    """

    def __init__(self, dim=None, scale="lin", **attrs):
        super().__init__(
            dim=dim, scale=scale, **attrs
        )


class AttrColor(AttrBase):

    """
    Color attribute.

    Parameters
    ----------
    dim : int
        Data dimension.
    scale : "const", "lin", "log"
        Color map scaling or static color.
    map : cfg.colors
        Color map string or static color.
    alpha : float
        Opacity.
    min, max : float
        Minimum, maximum of color normalization.
    """

    def __init__(self, dim=None, scale="const", map=None, alpha=1,
                 min=None, max=None, **attrs):
        super().__init__(
            dim=dim, scale=scale, map=map, alpha=alpha,
            min=min, max=max, **attrs
        )


class AttrSize(AttrBase):

    """
    Size scaling attribute.

    Parameters
    ----------
    xdim, ydim, zdim : int
        Data dimension for scaling in (x, y, z) plot direction.
    xscale, yscale, zscale : "const", "lin", "log"
        Size scaling type or static size.
    xmap, ymap, zmap : float
        Numeric scale or static size.
    """

    def __init__(self,
                 xdim=None, ydim=None, zdim=None,
                 xscale="const", yscale="const", zscale="const",
                 xmap=None, ymap=None, zmap=None, **attrs):
        super().__init__(
            xdim=xdim, ydim=ydim, zdim=zdim,
            xscale=xscale, yscale=yscale, zscale=zscale,
            xmap=xmap, ymap=ymap, zmap=zmap, **attrs
        )


class AttrLine(AttrBase):

    """
    Line style attribute.

    Parameters
    ----------
    shape : "-", "--", "-.", ":"
        Line style: continuous, dashed, dot-dashed, dotted.
    thickness : float
        Line thickness.
    """

    def __init__(self, shape=None, thickness=None, **attrs):
        super().__init__(shape=shape, thickness=thickness, **attrs)


class AttrFill(AttrBase):

    """
    TODO: Point/line/surface fill attribute.
    """

    def __init__(self, **attrs):
        super().__init__(**attrs)


# ++++++++++++++++++++++++


class AttrPoint(object):

    """
    Plot type: point plot.
    0D objects embedded in 2D/3D plot.

    Parameters
    ----------
    xpos, ypos, zpos : AttrPosition
        Plot value position.
    color : AttrColor
        Color of point.
    shape : matplotlib.markers
        Point type.
    size : AttrSize
        Point size.
    fill : AttrFill
        Fill to point.
    """

    def __init__(self,
                 xpos=None, ypos=None, zpos=None,
                 color=None, shape=None, size=None, fill=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.zpos = misc.assume_construct_obj(zpos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.shape = shape
        self.size = misc.assume_construct_obj(size, AttrSize)
        self.fill = misc.assume_construct_obj(fill, AttrFill)


class AttrCurve(object):

    """
    Plot type: curve plot.
    1D object embedded in 2D/3D plot.

    Parameters
    ----------
    xpos, ypos, zpos : AttrPosition
        Plot value position.
    color : AttrColor
        Color of curve.
    line : AttrLine
        Line style.
    fill : AttrFill
        Fill to line.
    """

    def __init__(self,
                 xpos=None, ypos=None, zpos=None,
                 color=None, line=None, fill=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.zpos = misc.assume_construct_obj(zpos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.line = misc.assume_construct_obj(line, AttrLine)
        self.fill = misc.assume_construct_obj(fill, AttrFill)


class AttrMatrix(object):

    """
    Plot type: matrix (color) plot.
    2D object embedded in 2D plot.

    Parameters
    ----------
    xpos, ypos : AttrPosition
        Plot value position.
    color : AttrColor
        Color of matrix fill.
    meshcolor : AttrColor
        Color of mesh.
    """

    def __init__(self,
                 xpos=None, ypos=None, color=None,
                 mesh=None, meshcolor=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.meshcolor = misc.assume_construct_obj(meshcolor, AttrColor)


class AttrSurface(object):

    """
    Plot type: surface plot.
    2D object embedded in 3D plot.

    Parameters
    ----------
    xpos, ypos, zpos : AttrPosition
        Plot value position.
    color : AttrColor
        Color of surface.
    mesh : AttrLine
        Mesh (i.e. grid) lines.
    meshcolor : AttrColor
        Color of mesh.
        Color maps can only be used when the
        surface is transparent. With finite
        surface opacity, only static colors
        are allowed.
    fill : AttrFill
        Fill to surface.

    Notes
    -----
    Does not support partial transparency.
    """

    def __init__(self,
                 xpos=None, ypos=None, zpos=None, color=None,
                 mesh=None, meshcolor=None, fill=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.zpos = misc.assume_construct_obj(zpos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.mesh = misc.assume_construct_obj(mesh, AttrLine)
        self.meshcolor = misc.assume_construct_obj(meshcolor, AttrColor)
        self.fill = misc.assume_construct_obj(fill, AttrFill)


class AttrContour(object):

    """
    Plot type: contour plot.
    2D/3D object embedded in 2D/3D plot.

    Parameters
    ----------
    xpos, ypos, zpos : AttrPosition
        Plot value position.
    color : AttrColor
        Color of contours.
    levels : int
        Number of contour colors.
    """

    def __init__(self,
                 xpos=None, ypos=None, zpos=None, color=None, levels=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.zpos = misc.assume_construct_obj(zpos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.levels = levels


###############################################################################


class PlotCfg(object):

    """
    Parameters
    ----------
    xgridspec, ygridspec : tuple(int) or int
        Matplotlib gridspec usage (row, column).
        Scalars represent a single grid item,
        tuples are interpreted as slices.
    point : list(AttrPoint)
        Point plots.
    curve : list(AttrCurve)
        Curve plots.
    matrix : list(AttrMatrix)
        Matrix plots.
    surface : list(AttrSurface)
        Surface plots.
    contour : list(AttrContour)
        Contour plots.

    Notes
    -----
    2D (2D point, 2D curve, matrix, 2D contour) and 3D (3D point, 3D curve,
    surface, 3D contour) plots are incompatible to each other. Specifying both
    leads to undefined behaviour.
    """

    def __init__(self,
                 xgridspec=1, ygridspec=1,
                 point=None, curve=None, matrix=None,
                 surface=None, contour=None):
        self.xgridspec = xgridspec
        self.ygridspec = ygridspec
        self.point = misc.assume_construct_obj(point, AttrPoint)
        self.curve = misc.assume_construct_obj(curve, AttrCurve)
        self.matrix = misc.assume_construct_obj(matrix, AttrMatrix)
        self.surface = misc.assume_construct_obj(surface, AttrSurface)
        self.contour = misc.assume_construct_obj(contour, AttrContour)

    def get_gridspec_param(self):
        """
        Gets parameters for the `matplotlib.gridspec.GridSpec.new_subplotspec`
        method.

        Returns
        -------
        loc : tuple(int)
            (x, y) location of subplotspec.
        span : tuple(int)
            (xspan, yspan) span of subplotspec.

        Notes
        -----
        GridSpec coordinates count from left-top, thus x-axis (y-axis)
        corresponds to rows (columns).
        """
        xloc, xspan = (
            (self.xgridspec, 1) if np.isscalar(self.xgridspec)
            else (self.xgridspec[0], self.xgridspec[1] - self.xgridspec[0])
        )
        yloc, yspan = (
            (self.ygridspec, 1) if np.isscalar(self.ygridspec)
            else (self.ygridspec[0], self.ygridspec[1] - self.ygridspec[0])
        )
        return (xloc, yloc), (xspan, yspan)

    def get_plot_dim(self):
        """
        Gets the plot dimension (2D or 3D canvas).

        Returns
        -------
        dim : 2 or 3 or None
            Plot dimension. None if no plot is specified.

        Notes
        -----
        Assumes correctness of configuration, particularly that dimensions are
        consistent across plot types.
        """
        dim = None
        if self.point is not None:
            dim = 2 if self.point.zpos is None else 3
        elif self.curve is not None:
            dim = 2 if self.curve.zpos is None else 3
        elif self.matrix is not None:
            dim = 2
        elif self.surface is not None:
            dim = 3
        elif self.contour is not None:
            dim = 2 if self.contour.zpos is None else 3
        return dim


# ++++++++++++++++++++++++


def _plot_meta(mpl_ax, plot_dim, cfg, data):
    if isinstance(data, arraydata.ArrayData):
        mpl_ax.set_xlabel(data.scale.quantity[0].mathstr())
        mpl_ax.set_ylabel(data.scale.quantity[1].mathstr())
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
            xlabel = data.values[cfg.point.xpos.dim]
            ylabel = data.values[cfg.point.ypos.dim]
            if plot_dim == 3:
                zlabel = data.values[cfg.point.zpos.dim]
        elif cfg.matrix is not None:
            xlabel = data.values[cfg.matrix.xpos.dim]
            ylabel = data.values[cfg.matrix.ypos.dim]
            if plot_dim == 3:
                zlabel = data.values[cfg.matrix.zpos.dim]
        elif cfg.contour is not None:
            xlabel = data.values[cfg.contour.xpos.dim]
            ylabel = data.values[cfg.contour.ypos.dim]
            if plot_dim == 3:
                zlabel = data.values[cfg.contour.zpos.dim]
        elif cfg.surface is not None:
            xlabel = data.values[cfg.surface.xpos.dim]
            ylabel = data.values[cfg.surface.ypos.dim]
            if plot_dim == 3:
                zlabel = data.values[cfg.surface.zpos.dim]
        if xlabel is not None:
            mpl_ax.set_xlabel(xlabel)
        if ylabel is not None:
            mpl_ax.set_ylabel(ylabel)
        if zlabel is not None:
            mpl_ax.set_ylabel(zlabel)


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


def _plot_data_array(mpl_ax, plot_dim, cfg, x, y, data):
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
    """
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
            mpl_ax.scatter(
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
                c = np.full_like(x, 0)
                alpha = 0
            if (cfg.matrix.meshcolor is not None
                    and cfg.matrix.meshcolor.scale == "const"):
                edgecolors = cfg.matrix.meshcolor.cmap   # only static colors
            mpl_ax.pcolormesh(
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
                z = np.full_like(x, 0)
                alpha = 0
            levels = cfg.contour.levels
            mpl_ax.contourf(
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
            mpl_ax.scatter(
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
            if cfg.point.color is not None and cfg.point.color.dim is not None:
                _color = _param_color(cfg.point.color, data)
                cdata, color, cmap, vmin, vmax, alpha = _color
                if (cfg.point.meshcolor is not None
                        and cfg.point.meshcolor.scale == "const"):
                    mcolor = cfg.point.meshcolor.map
            # Wireframe plot
            else:
                if (cfg.point.meshcolor is not None
                        and cfg.point.meshcolor.dim is not None):
                    _color = _param_color(cfg.point.meshcolor, data)
                    cdata, _, cmap, _, _, _ = _color
            # Surface plot
            if alpha != 0:
                mpl_ax.plot_surface(
                    xgrid, ygrid, z, color=color, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolor=mcolor
                )
            # Wireframe plot
            else:
                cdata = (cdata - cdata.min()) / (cdata.max() - cdata.min())
                surf = mpl_ax.plot_surface(
                    xgrid, ygrid, z, facecolors=cdata
                )
                surf.set_facecolors((0, 0, 0, 0))
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
            mpl_ax.contour(
                xgrid, ygrid, z, levels=levels, alpha=alpha, cmap=cmap,
                vmin=vmin, vmax=vmax
            )


def _plot_data_series(mpl_ax, plot_dim, cfg, data):
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
            mpl_ax.scatter(
                xx, yy, s=s, c=c, marker=marker, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha,
            )
        # 2D curve line plot
        if cfg.curve is not None:
            if cfg.curve.xpos is None or cfg.curve.ypos is None:
                return
            xx, yy = data[cfg.point.xpos.dim], data[cfg.point.ypos.dim]
            c, alpha, linestyle, linewidth = 4 * [None]
            if (cfg.point.color is not None
                    and cfg.point.color.scale == "const"):
                c = cfg.point.color.map
                alpha = cfg.point.color.alpha
            if cfg.point.line is not None:
                linestyle = cfg.curve.line.shape
                linewidth = cfg.curve.line.thickness
            mpl_ax.plot(
                xx, yy, color=c, alpha=alpha,
                linestyle=linestyle, linewidth=linewidth
            )
        # 2D color matrix plot
        if cfg.matrix is not None:
            if cfg.matrix.xpos is None or cfg.matrix.ypos is None:
                return
            xx, yy = data[cfg.matrix.xpos.dim], data[cfg.matrix.ypos.dim]
            c, cmap, vmin, vmax, alpha = 5 * [None]
            if cfg.matrix.color is not None:
                c = data[cfg.point.color.dim]
                cmap = cfg.point.color.map
                vmin, vmax = cfg.point.color.min, cfg.point.color.max
                alpha = cfg.point.color.alpha
            else:
                c = np.full_like(xx, 0)
                alpha = 0
            mpl_ax.tripcolor(
                xx, yy, c, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha
            )
            if (cfg.matrix.meshcolor is not None
                    and cfg.matrix.meshcolor.scale == "const"):
                linestyle = cfg.matrix.mesh.shape
                linewidth = cfg.matrix.mesh.thickness
                mpl_ax.triplot(
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
            mpl_ax.tricontour(
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
            mpl_ax.scatter(
                xx, yy, zz, s=s, c=c, marker=marker, cmap=cmap,
                vmin=vmin, vmax=vmax, alpha=alpha,
            )
        # 3D curve line plot
        if cfg.curve is not None:
            if (cfg.curve.xpos is None or cfg.curve.ypos is None
                    or cfg.curve.zpos is None):
                return
            xx, yy = data[cfg.point.xpos.dim], data[cfg.point.ypos.dim]
            zz = data[cfg.point.zpos.dim]
            c, alpha, linestyle, linewidth = 4 * [None]
            if (cfg.point.color is not None
                    and cfg.point.color.scale == "const"):
                c = cfg.point.color.map
                alpha = cfg.point.color.alpha
            if cfg.point.line is not None:
                linestyle = cfg.curve.line.shape
                linewidth = cfg.curve.line.thickness
            mpl_ax.plot(
                xx, yy, zz, color=c, alpha=alpha,
                linestyle=linestyle, linewidth=linewidth
            )
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
                mpl_ax.plot_trisurf(
                    xx, yy, zz, color=color, cmap=cmap,
                    vmin=vmin, vmax=vmax, edgecolor=mcolor
                )
            # Wireframe plot
            else:
                cdata = (cdata - cdata.min()) / (cdata.max() - cdata.min())
                surf = mpl_ax.plot_trisurf(
                    xx, yy, zz, facecolors=cdata
                )
                surf.set_facecolors((0, 0, 0, 0))
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
            mpl_ax.tricontour(
                xx, yy, z, levels=levels, alpha=alpha, cmap=cmap,
                vmin=vmin, vmax=vmax
            )


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
    _data_type_hint : "array" or "series"
        Hint for ambiguous data types whether to interpret
        data as array or series, e.g. for 2D numpy arrays.
    """
    data_type = None
    x, y = None
    # Distinguish data types
    if isinstance(data, arraydata.ArrayData):
        data_type = "array"
        _plot_meta(mpl_ax, plot_dim, cfg, data)
        data = data.data
        x, y = (
            np.linspace(data.scale.offset[0], data.scale.max[0],
                        num=data.data.shape[0]),
            np.linspace(data.scale.offset[1], data.scale.max[1],
                        num=data.data.shape[1])
        )
    elif isinstance(data, seriesdata.SeriesData):
        data_type = "series"
        _plot_meta(mpl_ax, plot_dim, cfg, data)
    elif isinstance(data, list):
        data_type = "series"
    elif isinstance(data, np.ndarray):
        if len(data.shape) == 1:
            data_type = "series"
        elif len(data.shape) > 2:
            data_type = "array"
        else:
            data_type = _data_type_hint
    # Call plot functions
    if data_type == "array":
        _plot_data_array(mpl_ax, plot_dim, cfg, x, y, data)
    elif data_type == "series":
        _plot_data_series(mpl_ax, plot_dim, cfg, data)


###############################################################################


class FigureCfg(object):

    """
    Figure configuration.

    Parameters
    ----------
    hrzt_size, vert_size : int
        Horizontal and vertical figure size in points (pt).
    resolution : float
        Resolution in dots per point (1/pt).
    format_ : str
        Plot format when saved.
    """

    def __init__(self,
                 hrzt_size=None, vert_size=None, resolution=2.0,
                 format_="pdf"):
        self.hrzt_size = hrzt_size
        self.vert_size = vert_size
        self.resolution = resolution
        self.format = format_

    def get_size(self, unit="pt"):
        """
        Gets the (horizontal, vertical) size.

        Parameters
        ----------
        unit : "pt", "mm", "cm", "in"
            Unit in points, millimeter, centimeter,
            meter, inch.
        """
        if self.hrzt_size is None or self.vert_size is None:
            return None
        else:
            cv_factor = 1
            if unit == "in":
                cv_factor = 72
            elif unit == "mm":
                cv_factor = 72 / 25.4
            elif unit == "cm":
                cv_factor = 72 / 2.54
            return (self.hrzt_size * cv_factor, self.vert_size * cv_factor)

    def get_resolution(self, unit="pt"):
        if self.resolution is None:
            return None
        else:
            cv_factor = 1
            if unit == "in":
                cv_factor = 1 / 72
            elif unit == "mm":
                cv_factor = 25.4 / 72
            elif unit == "cm":
                cv_factor = 2.54 / 72
            return self.resolution * cv_factor


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
        self.figure_cfg = misc.assume_construct_obj(figure_cfg, FigureCfg)
        self.plot_cfgs = misc.assume_list(plot_cfgs)
        self.plot_dim = {}
        self.plot_style_cfg = None
        self.mpl_fig = None
        self.mpl_gs = None
        self.mpl_ax = {}
        self.mpl_ax_loc = []
        self.data = misc.assume_list(data)

    def add_mpl_ax(self, loc, xspan=None, yspan=None, plot_dim=None):
        """
        Adds an internal reference to already created axes to avoid
        matplotlib axes-reuse deprecation warning.

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
        elif plot_dim != self.plot_dim[loc]:
            raise ValueError("incompatible plot dimensions")

    def setup_mpl(self):
        """
        Sets up the matplotlib figure environment to enable plotting.
        """
        # Create matplotlib figure
        self.mpl_fig = plt.figure(
            figsize=self.figure_cfg.get_size(unit="in"),
            dpi=self.figure_cfg.get_resolution(unit="in")
        )
        # Create matplotlib gridspec
        xmax, ymax = 0, 0
        for item in self.plot_cfgs:
            if np.isscalar(item.xgridspec):
                xmax = np.max(xmax, item.xgridspec + 1)
            else:
                xmax = np.max(xmax, *item.xgridspec)
            if np.isscalar(item.ygridspec):
                ymax = np.max(ymax, item.ygridspec + 1)
            else:
                ymax = np.max(ymax, *item.ygridspec)
        self.mpl_gs = mpl.gridspec.GridSpec(xmax, ymax, figure=self.mpl_fig)
        # Create matplotlib axes
        self.mpl_ax = len(self.plot_cfgs) * [None]
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
        for i, plot_cfg in enumerate(self.plot_cfgs):
            plot(
                self.mpl_ax[self.mpl_ax_loc[i]],
                self.plot_dim[self.mpl_ax_loc[i]],
                plot_cfg, self.data[i]
            )


###############################################################################


def _flatten_nested_dict(d, delim="."):
    """
    Flattens a nested dictionary into a top-level dictionary. Levels are
    separated by the specified delimiter.
    """
    for key, val in d.items():
        if isinstance(val, dict):
            for k, v in _flatten_nested_dict(val, delim=delim).items():
                d[key + delim + k] = v
            del d[key]
    return d


def cv_plotstylecfg_to_mplrc(cfg):
    """
    Converts a PlotStyleCfg to a matplotlib resource dictionary.
    """
    d = cfg.to_obj_dict()
    return _flatten_nested_dict(d, delim=".")


class PlotStyleBaseCfg(cfg.CfgBase):

    def __init__(self, **kwargs):
        super().__init__(group="plot_style")
        self.__dict__.update(kwargs)


class PlotStyleCfg(PlotStyleBaseCfg):

    """
    Implements the matplotlib resource file.

    For more information on parameters, refer to
    https://matplotlib.org/tutorials/introductory/customizing.html.
    """

    def __init__(
        self,
        lines={}, patches={}, hatches={},
        boxplot={}, font={}, text={}, mathtext={},
        axes={}, dates={}, xtick={}, ytick={}, grid={},
        legend={}, figure={}, images={}, contour={},
        errorbar={}, histogram={}, scatter={}, savefig={}
    ):
        super().__init__()
        self.lines = misc.assume_construct_obj(lines, PlotStyleBaseCfg)
        self.patches = misc.assume_construct_obj(patches, PlotStyleBaseCfg)
        self.hatches = misc.assume_construct_obj(hatches, PlotStyleBaseCfg)
        self.boxplot = misc.assume_construct_obj(boxplot, PlotStyleBoxplotCfg)
        self.font = misc.assume_construct_obj(font, PlotStyleBaseCfg)
        self.text = misc.assume_construct_obj(text, PlotStyleTextCfg)
        self.mathtext = misc.assume_construct_obj(mathtext, PlotStyleBaseCfg)
        self.axes = misc.assume_construct_obj(axes, PlotStyleAxesCfg)
        self.dates = misc.assume_construct_obj(dates, PlotStyleDateCfg)
        self.xtick = misc.assume_construct_obj(xtick, PlotStyleTickCfg)
        self.ytick = misc.assume_construct_obj(ytick, PlotStyleTickCfg)
        self.grid = misc.assume_construct_obj(grid, PlotStyleBaseCfg)
        self.legend = misc.assume_construct_obj(legend, PlotStyleBaseCfg)
        self.figure = misc.assume_construct_obj(figure, PlotStyleFigureCfg)
        self.images = misc.assume_construct_obj(images, PlotStyleBaseCfg)
        self.contour = misc.assume_construct_obj(contour, PlotStyleBaseCfg)
        self.errorbar = misc.assume_construct_obj(errorbar, PlotStyleBaseCfg)
        self.histogram = misc.assume_construct_obj(histogram, PlotStyleBaseCfg)
        self.scatter = misc.assume_construct_obj(scatter, PlotStyleBaseCfg)
        self.savefig = misc.assume_construct_obj(savefig, PlotStyleBaseCfg)


class PlotStyleBoxplotCfg(PlotStyleBaseCfg):

    def __init__(
        self,
        flierprops={}, boxprops={}, whiskersprops={},
        capprops={}, medianprops={}, meanprops={},
        **kwargs
    ):
        super().__init__(**kwargs)
        self.flierprops = misc.assume_construct_obj(flierprops,
                                                    PlotStyleBaseCfg)
        self.boxprops = misc.assume_construct_obj(boxprops, PlotStyleBaseCfg)
        self.whiskersprops = misc.assume_construct_obj(whiskersprops,
                                                       PlotStyleBaseCfg)
        self.capprops = misc.assume_construct_obj(capprops, PlotStyleBaseCfg)
        self.medianprops = misc.assume_construct_obj(medianprops,
                                                     PlotStyleBaseCfg)
        self.meanprops = misc.assume_construct_obj(meanprops, PlotStyleBaseCfg)


class PlotStyleTextCfg(PlotStyleBaseCfg):

    def __init__(self, latex={}, **kwargs):
        super().__init__(**kwargs)
        self.latex = misc.assume_construct_obj(latex, PlotStyleBaseCfg)


class PlotStyleAxesCfg(PlotStyleBaseCfg):

    def __init__(self, formatter={}, spines={}, prop_cycle=None, **kwargs):
        super().__init__(**kwargs)
        self.formatter = misc.assume_construct_obj(formatter, PlotStyleBaseCfg)
        self.spines = misc.assume_construct_obj(spines, PlotStyleBaseCfg)
        if prop_cycle is not None:
            if isinstance(prop_cycle, cycler.Cycler):
                self.prop_cycle = prop_cycle
            else:
                self.prop_cycle = cycler.cycler("color", prop_cycle)


class PlotStyleDateCfg(PlotStyleBaseCfg):

    def __init__(self, autoformatter={}):
        super().__init__()
        self.autoformatter = misc.assume_construct_obj(autoformatter,
                                                       PlotStyleBaseCfg)


class PlotStyleTickCfg(PlotStyleBaseCfg):

    def __init__(self, minor={}, major={}, **kwargs):
        super().__init__(**kwargs)
        self.minor = misc.assume_construct_obj(minor, PlotStyleBaseCfg)
        self.major = misc.assume_construct_obj(major, PlotStyleBaseCfg)


class PlotStyleFigureCfg(PlotStyleBaseCfg):

    def __init__(self, subplot={}, constrained_layout={}, **kwargs):
        super().__init__(**kwargs)
        self.subplot = misc.assume_construct_obj(subplot, PlotStyleBaseCfg)
        self.constrained_layout = misc.assume_construct_obj(
            constrained_layout, PlotStyleBaseCfg
        )
