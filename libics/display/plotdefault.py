from libics.display import plotcfg


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
    hrzt_subplot_pos=0, vert_subplot_pos=0, label=None
):
    curve_cfg = plotcfg.AttrCurve(
        ypos={"dim": -1, "scale": "lin"}
    )
    return plotcfg.PlotCfg(
        xgridspec=hrzt_subplot_pos, ygridspec=vert_subplot_pos,
        curve=curve_cfg, label=label
    )


def get_plotcfg_arraydata_2d(
    hrzt_subplot_pos=0, vert_subplot_pos=0, label=None, aspect=None
):
    matrix_cfg = plotcfg.AttrMatrix(
        color={"dim": -1, "scale": "lin", "map": "viridis", "alpha": 1}
    )
    return plotcfg.PlotCfg(
        xgridspec=hrzt_subplot_pos, ygridspec=vert_subplot_pos,
        matrix=matrix_cfg, aspect=aspect, label=label
    )


def get_plotcfg_seriesdata_1d(
    hrzt_subplot_pos=0, vert_subplot_pos=0, label=None
):
    curve_cfg = plotcfg.AttrCurve(
        xpos={"dim": 0, "scale": "lin"}, ypos={"dim": -1, "scale": "lin"}
    )
    return plotcfg.PlotCfg(
        xgridspec=hrzt_subplot_pos, ygridspec=vert_subplot_pos,
        curve=curve_cfg, label=label
    )


def get_plotcfg_seriesdata_2d(
    hrzt_subplot_pos=0, vert_subplot_pos=0, aspect=None, label=None
):
    matrix_cfg = plotcfg.AttrMatrix(
        xpos={"dim": 0, "scale": "lin"}, ypos={"dim": 1, "scale": "lin"},
        color={"dim": -1, "scale": "lin", "map": "viridis", "alpha": 1}
    )
    return plotcfg.PlotCfg(
        xgridspec=hrzt_subplot_pos, ygridspec=vert_subplot_pos,
        matrix=matrix_cfg, aspect=aspect, label=label
    )


###############################################################################


if __name__ == "__main__":
    # Test imports
    import matplotlib.pyplot as plt
    import numpy as np
    from libics import data
    from libics.display import plot

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
    fig = plot.Figure(
        fig_cfg,
        [sd_plot_cfg, ad_plot_cfg, sd2_plot_cfg, ad2_plot_cfg],
        data=[sd, ad, sd2, ad2]
    )
    fig.setup_mpl()
    fig.plot()
    fig.legend()
    plt.tight_layout()
    plt.show()
