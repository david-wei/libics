import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from libics import cfg
from libics.data import arraydata, seriesdata
from libics.util import misc


###############################################################################


def _plot_tuple(data, plot):
    pass


def _plot_list(data, plot):
    pass


def _plot_numpy_ndarray(data, plot):
    pass


def _plot_arraydata(data, plot):
    pass


def _plot_seriesdata(data, plot):
    pass


def plot(data, plot=None):
    """
    Plots the data according to the plot object.

    Parameters
    ----------
    data : list, tuple, numpy.ndarray, ArrayData, SeriesData
        Data to be plotted.
    plot : Plot
        Plot wrapper object.
    """
    mpl_rcc = None, None, None
    # Assert existing plot axes
    if plot is None:
        plot_cfg = PlotCfg()
        plot = Plot(plot_cfg, mpl_ax=plt.gca())
        plot.setup()
    if plot.ax is None:
        plot.setup()
    # Enable style
    if plot.mpl_rc is not None:
        mpl_rcc = mpl.rc_context(rc=plot.mpl_rc)
    # Perform plot
    if isinstance(data, seriesdata.SeriesData):
        _plot_seriesdata(data, plot)
    elif isinstance(data, arraydata.ArrayData):
        _plot_arraydata(data, plot)
    elif isinstance(data, np.ndarray):
        _plot_numpy_ndarray(data, plot)
    elif isinstance(data, list):
        _plot_list(data, plot)
    elif isinstance(data, tuple):
        _plot_tuple(data, plot)
    # Disable style
    if mpl_rcc is not None:
        mpl_rcc.__exit__(None, None, None)


def plot_multi(data_ls, plot_ls=None):
    """
    Plots multiple data sets.
    """
    if plot_ls is None:
        plot_ls = len(data_ls) * [None]
    assert(len(data_ls) == len(plot_ls))
    for i, data in enumerate(data_ls):
        plot(data, plot=plot_ls[i])


###############################################################################


class FigCfg(cfg.CfgBase):

    """
    Matplotlib figure configuration.

    Parameters
    ----------
    gridspec_rows : int
        Rows of matplotlib figure's gridspec.
    gridspec_cols : int
        Columns of matplotlib figure's gridspec.
    spacing_rows : float or None
        Vertical spacing between subplots in
        fractions of average axis height.
    spacing_cols : float or None
        Horiontal spacing between subplots in
        fractions of average axis width.
    """

    def __init__(self, gridspec_rows=1, gridspec_cols=1,
                 spacing_rows=None, spacing_cols=None):
        super().__init__(group="plot_configuration")
        self.gridspec_rows = gridspec_rows
        self.gridspec_cols = gridspec_cols
        self.spacing_rows = spacing_rows
        self.spacing_cols = spacing_cols


class Figure(object):

    """
    Parameters
    ----------
    fig_cfg : FigCfg
        Figure configuration.
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    """

    def __init__(self, fig_cfg, mpl_fig=None):
        self.fig_cfg
        self.mpl_fig = mpl_fig
        self.gridspec = None

    def setup(self):
        """
        Sets up matplotlib figure and gridspec.
        """
        if self.mpl_fig is None:
            self.mpl_fig = plt.figure()
        self.gridspec = mpl.gridspec.GridSpec(
            self.fig_cfg.gridspec_rows, self.fig_cfg.gridspec_cols,
            wspace=self.fig_cfg.spacing_cols, hspace=self.fig_cfg.spacing_rows
        )


class PLOT_TYPE:

    DEFAULT = 0
    NONE = 10
    LINE = 101
    SCATTER = 102
    COLOR = 103
    BAR = 104


class PlotCfg(cfg.CfgBase):

    """
    Matplotlib axes configuration.

    Parameters
    ----------
    plot_type : PLOT_TYPE or list(PLOT_TYPE)
        Plot type for each dimension of data.
    plot_style : PlotStyleCfg or None
        Matplotlib plot style configuration.
        If None, default rc_params are chosen.
    plot_xlayout : tuple(int) or int
        Matplotlib gridspec usage (row slice).
    plot_ylayout : tuple(int) or int
        Matplotlib gridspec usage (column slice).
    """

    def __init__(self, plot_type=PLOT_TYPE.DEFAULT, plot_style=None,
                 plot_xlayout=1, plot_ylayout=1):
        super().__init__(group="plot_configuration")
        self.plot_type = plot_type
        self.plot_style = misc.assume_construct_obj(plot_style, PlotStyleCfg)
        self.plot_xlayout = plot_xlayout
        self.plot_ylayout = plot_ylayout


class Plot(object):

    """
    Parameters
    ----------
    plot_cfg : PlotCfg
        Plot configuration.
    figure : Figure
        Figure wrapper object.
    """

    def __init__(self, plot_cfg, figure=None, mpl_ax=None):
        self.plot_cfg = plot_cfg
        self.mpl_ax = mpl_ax
        self.figure = figure
        self.mpl_rc = None

    def setup(self):
        """
        Sets up matplotlib axes and style dictionary.
        """
        xlayout = self.plot_cfg.plot_xlayout
        ylayout = self.plot_cfg.plot_ylayout
        xmax, ymax = None, None
        if isinstance(xlayout, tuple):
            xmax = max(xlayout)
            xlayout = slice(*xlayout)
        if isinstance(ylayout, tuple):
            ymax = max(ylayout)
            ylayout = slice(*ylayout)
        if self.figure is None:
            fig_cfg = FigCfg(gridspec_rows=ymax, gridspec_cols=xmax)
            self.figure = Figure(fig_cfg, mpl_fig=plt.gcf())
            self.figure.setup()
            if self.mpl_ax is None:
                self.mpl_ax = self.figure.fig.add_subplot(
                    self.figure.gridspec[xlayout, ylayout]
                )
        else:
            self.mpl_ax = self.figure.fig.add_subplot(
                self.figure.gridspec[xlayout, ylayout]
            )
        self.mpl_rc = cv_plotstylecfg_to_mplrc(self.plot_cfg.plot_style)


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
