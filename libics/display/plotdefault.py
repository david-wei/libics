import matplotlib as mpl
import numpy as np

from libics import display
from libics.display import plotcfg
from libics.data import arraydata, seriesdata
from libics.util import misc


###############################################################################


def get_figurecfg(
    hrzt_subplot_count=1, vert_subplot_count=1, format_="pdf"
):
    return plotcfg.FigureCfg(
        hrzt_size_scale=hrzt_subplot_count, vert_size_scale=vert_subplot_count,
        resolution=4.0, format_=format_
    )


###############################################################################


def get_plotcfg_arraydata_1d(
    hrzt_subplot_pos=0, vert_subplot_pos=0, label=None, color=None, alpha=1
):
    curve_cfg = plotcfg.AttrCurve(
        ypos={"dim": -1, "scale": "lin"},
        color={"map": color, "alpha": alpha}
    )
    return plotcfg.PlotCfg(
        xgridspec=vert_subplot_pos, ygridspec=hrzt_subplot_pos,
        curve=curve_cfg, label=label
    )


def get_plotcfg_arraydata_2d(
    hrzt_subplot_pos=0, vert_subplot_pos=0,
    label=None, aspect=None, color="viridis", alpha=1, min=None, max=None,
    colorbar=True
):
    matrix_cfg = plotcfg.AttrMatrix(
        color={"dim": -1, "scale": "lin", "map": color, "alpha": alpha,
               "min": min, "max": max, "colorbar": colorbar}
    )
    return plotcfg.PlotCfg(
        xgridspec=vert_subplot_pos, ygridspec=hrzt_subplot_pos,
        matrix=matrix_cfg, aspect=aspect, label=label
    )


def get_plotcfg_seriesdata_1d(
    hrzt_subplot_pos=0, vert_subplot_pos=0, label=None, color=None, alpha=1,
    plot_type="line", linestyle=None, point_size=None, edgecolor=None, marker="o"
):
    """
    plot_type : str
        "line", "point", "yerr",
    """
    curve_cfg, point_cfg = None, None
    if plot_type == "line":
        curve_cfg = plotcfg.AttrCurve(
            xpos={"dim": 0, "scale": "lin"}, ypos={"dim": -1, "scale": "lin"},
            color={"map": color, "alpha": alpha}, line={"shape": linestyle}
        )
    else:
        if point_size is None:
            point_size = float(mpl.rcParams["font.size"]) / 2
        edgesize = 0.18 * point_size
        if edgecolor is None:
            edgecolor = color
        yerr = None
        if plot_type == "yerr":
            yerr = {"dim": 2}
        point_cfg = plotcfg.AttrPoint(
            xpos={"dim": 0, "scale": "lin"}, ypos={"dim": 1, "scale": "lin"},
            color={"map": color, "alpha": alpha}, shape=marker,
            edgecolor={"map": edgecolor, "alpha": alpha, "scale": "const"},
            size={"xmap": point_size}, edgesize=edgesize, yerr=yerr
        )
    return plotcfg.PlotCfg(
        xgridspec=vert_subplot_pos, ygridspec=hrzt_subplot_pos,
        curve=curve_cfg, point=point_cfg, label=label
    )


def get_plotcfg_seriesdata_2d(
    hrzt_subplot_pos=0, vert_subplot_pos=0,
    aspect=None, label=None, color="viridis", alpha=1, colorbar=True
):
    matrix_cfg = plotcfg.AttrMatrix(
        xpos={"dim": 0, "scale": "lin"}, ypos={"dim": 1, "scale": "lin"},
        color={"dim": -1, "scale": "lin", "map": color, "alpha": alpha,
               "min": min, "max": max, "colorbar": True}
    )
    return plotcfg.PlotCfg(
        xgridspec=vert_subplot_pos, ygridspec=hrzt_subplot_pos,
        matrix=matrix_cfg, aspect=aspect, label=label
    )


###############################################################################


def _get_pcfg(data, hrzt_subplot_pos=0, vert_subplot_pos=0,
              label=None, **kwargs):
    func = None
    if isinstance(data, arraydata.ArrayData):
        if data.scale.get_dim() <= 2:
            func = get_plotcfg_arraydata_1d
        else:
            func = get_plotcfg_arraydata_2d
    elif isinstance(data, seriesdata.SeriesData):
        if data.get_dim() <= 2:
            func = get_plotcfg_seriesdata_1d
        else:
            func = get_plotcfg_seriesdata_2d
    return func(
        hrzt_subplot_pos=hrzt_subplot_pos, vert_subplot_pos=vert_subplot_pos,
        label=label, **kwargs
    )


def _flatten_data(data, label=None):
    labels, data_flat = [], []
    if isinstance(data, list) or isinstance(data, tuple):
        for item in data:
            l, df = _flatten_data(item)
            labels += l
            data_flat += df
    elif isinstance(data, dict):
        for key, val in data.items():
            l, df = _flatten_data(val, label=key)
            labels += l
            data_flat += df
    else:
        labels.append(label)
        data_flat.append(data)
    return labels, data_flat


def plot(*data, **kwargs):
    labels, data = _flatten_data(data)
    ratio = 3 / np.sqrt(3**2 + 2**2)
    hgrid = int(np.ceil(np.sqrt(len(labels) / ratio)))
    vgrid = int(np.ceil(len(labels) / hgrid))
    spos = misc.get_combinations((range(hgrid), range(vgrid)))[:len(labels)]
    pcfgs = [_get_pcfg(d, hrzt_subplot_pos=spos[i][0],
                       vert_subplot_pos=spos[i][1], label=labels[i], **kwargs)
             for i, d in enumerate(data)]
    fcfg = get_figurecfg(hrzt_subplot_count=hgrid, vert_subplot_count=vgrid)
    fig = display.plot.Figure(fcfg, pcfgs, data=data)
    fig.plot()
    fig.legend()
    return fig


###############################################################################


if __name__ == "__main__":
    # Test imports
    import matplotlib.pyplot as plt
    from libics import data

    # Test data
    x = np.arange(200) - 50
    y = np.arange(200)**2
    xx, yy = np.linspace(-2, 2, 201), np.linspace(-2, 2, 200)
    xgrid, ygrid = np.meshgrid(xx, yy)
    zgrid = np.exp(-(xgrid / 2)**2 - ygrid**2)

    # 1D SeriesData
    sd = data.seriesdata.SeriesData()
    sd.add_dim(name="xpos", symbol="x", unit="mm")
    sd.add_dim(name="ypos", symbol="y", unit="mm")
    sd.data = [x, y]
    # 1D ArrayData
    ad = data.arraydata.ArrayData()
    ad.scale.add_dim(offset=x[0], scale=((x[-1] - x[0]) / len(x)),
                     name="xpos", symbol="x", unit="mm")
    ad.scale.add_dim(name="ypos", symbol="y", unit="mm")
    ad.data = y
    ad.set_max()
    # 2D SeriesData
    sd2 = data.seriesdata.SeriesData()
    sd2.add_dim(name="pos", symbol="x", unit="mm")
    sd2.add_dim(name="pressure", symbol="p = \\frac{F}{A}", unit="bar")
    sd2.add_dim(name="temperature", symbol="T", unit="K")
    sd2.data = [xgrid.flatten(), ygrid.flatten(), zgrid.flatten()]
    # 2D ArrayData
    ad2 = data.arraydata.ArrayData()
    ad2.scale.add_dim(offset=xx[0], scale=((xx[-1] - xx[0]) / len(xx)),
                      name="pos", symbol="x", unit="mm")
    ad2.scale.add_dim(offset=yy[0], scale=((yy[-1] - yy[0]) / len(yy)),
                      name="pressure", symbol="p", unit="bar")
    ad2.scale.add_dim(name="temperature", symbol="T", unit="K")
    ad2.data = zgrid
    ad2.set_max()

    # Create figure/plot configuration
    fig_cfg = get_figurecfg(2, 2)
    sd_plot_cfg = get_plotcfg_seriesdata_1d(0, 0, "sd")
    ad_plot_cfg = get_plotcfg_arraydata_1d(0, 1, "ad")
    sd2_plot_cfg = get_plotcfg_seriesdata_2d(1, 0, "sd2")
    ad2_plot_cfg = get_plotcfg_arraydata_2d(1, 1, "ad2")
    # Plot
    fig = display.plot.Figure(
        fig_cfg,
        [sd_plot_cfg, ad_plot_cfg, sd2_plot_cfg, ad2_plot_cfg],
        data=[sd, ad, sd2, ad2]
    )
    fig.setup_mpl()
    fig.plot()
    fig.legend()
    plt.tight_layout()
    plt.show()
