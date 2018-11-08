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
        Color map string.
    alpha : float
        Opacity.
    min, max : float
        Minimum, maximum of color normalization.
    """

    def __init__(self, dim=None, scale="const", map=None, alpha=1, **attrs):
        super().__init__(
            dim=dim, scale=scale, map=map, alpha=alpha, **attrs
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
            self, xdim=xdim, ydim=ydim, zdim=zdim,
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
    """

    def __init__(self,
                 xpos=None, ypos=None, zpos=None,
                 color=None, shape=None, size=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.zpos = misc.assume_construct_obj(zpos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.shape = shape
        self.size = misc.assume_construct_obj(size, AttrSize)


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
    """

    def __init__(self,
                 xpos=None, ypos=None, zpos=None,
                 color=None, line=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.zpos = misc.assume_construct_obj(zpos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.line = misc.assume_construct_obj(line, AttrLine)


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
    mesh : AttrLine
        Mesh (i.e. grid) lines.
    meshcolor : AttrColor
        Color of mesh.
    """

    def __init__(self,
                 xpos=None, ypos=None, color=None,
                 mesh=None, meshcolor=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.mesh = misc.assume_construct_obj(mesh, AttrLine)
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
    """

    def __init__(self,
                 xpos=None, ypos=None, zpos=None, color=None,
                 mesh=None, meshcolor=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.zpos = misc.assume_construct_obj(zpos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)
        self.mesh = misc.assume_construct_obj(mesh, AttrLine)
        self.meshcolor = misc.assume_construct_obj(meshcolor, AttrColor)


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
    """

    def __init__(self,
                 xpos=None, ypos=None, zpos=None, color=None):
        self.xpos = misc.assume_construct_obj(xpos, AttrPosition)
        self.ypos = misc.assume_construct_obj(ypos, AttrPosition)
        self.zpos = misc.assume_construct_obj(zpos, AttrPosition)
        self.color = misc.assume_construct_obj(color, AttrColor)


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


# ++++++++++++++++++++++++


def _plot_meta():
    pass


def _plot_data():
    pass


def _plot(cfg, data):
    pass


class Plot(object):

    """
    TODO: Implement plotting logic.

    Parameters
    ----------
    cfg : list(PlotCfg)
        Plot configurations.
    data : list(object)
        Corresponding data to be plotted.
    """

    def __init__(self, cfg, data):
        self.cfg = cfg
        self.data = data

    def plot(self):
        pass


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
    """

    def __init__(self,
                 figure_cfg, plot_cfgs, plot_style_cfg=None):
        self.figure_cfg = misc.assume_construct_obj(figure_cfg, FigureCfg)
        self.plot_cfgs = misc.assume_list(plot_cfgs)
        self.plots = []
        self.plot_style_cfg = None
        self.mpl_fig = None
        self.mpl_gs = None
        self.mpl_ax = []

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
            self.mpl_ax[i] = mpl.subplot(
                self.mpl_gs.new_subplotspec(loc, *span)
            )

    def plot(self):
        pass


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
