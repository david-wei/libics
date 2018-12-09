import copy

import numpy as np
from scipy import interpolate

from libics.data import arraydata, seriesdata


###############################################################################


def cv_arraydata_to_seriesdata(ad):
    """
    Converts a data.arraydata.ArrayData object to a data.seriesdata.SeriesData
    object.

    Parameters
    ----------
    ad : data.arraydata.ArrayData
        Array data to be converted.

    Returns
    -------
    sd : data.seriesdata.SeriesData
        Converted series data.
    """
    mesh = np.indices(ad.data.shape).astype(float)
    dims = mesh.shape[0]
    for i in range(dims):
        mesh[i] = ad.scale.offset[i] + ad.scale.scale[i] * mesh[i]
    sd = seriesdata.SeriesData()
    for q in ad.scale.quantity:
        sd.add_dim(quantity=q)
    mesh = mesh.reshape((dims, -1))
    data_mesh = np.append(
        mesh,
        np.expand_dims(ad.data.flatten(), axis=0),
        axis=0
    )
    sd.data = data_mesh
    return sd


def cv_seriesdata_to_arraydata(sd, sampling_shape=None,
                               algorithm="cubic", fill=np.nan):
    """
    Converts a data.arraydata.ArrayData object to a data.seriesdata.SeriesData
    object.

    Parameters
    ----------
    sd : data.seriesdata.SeriesData
        Series data to be converted.
    sampling_shape : tuple(int)
        Array shape of interpolated data.
    algorithm : str
        Interpolation algorithm.
        "nearest":
            Nearest point.
        "linear":
            Linear interpolation.
        "cubic":
            Cubic interpolation (up to 2D).
    fill : float
        Fill value for linear and cubic interpolation if value
        is outside convex hull (i.e. needs extrapolation).

    Returns
    -------
    ad : data.arraydata.ArrayData
        Converted array data.
    """
    # Settings
    dims = sd.get_dim() - 1
    if dims > 2 and algorithm == "cubic":
        algorithm = "linear"
    mins, maxs = sd.data.min(axis=-1), sd.data.max(axis=-1)
    if sampling_shape is None:
        # TODO: proper sampling shape determination
        sampling_shape = dims * [round(2 * sd.data[:-1].size**(1 / dims))]
    elif isinstance(sampling_shape, int):
        sampling_shape = dims * [sampling_shape]
    # Set array scale
    ad = arraydata.ArrayData()
    for i, s in enumerate(sampling_shape):
        offset = mins[i]
        scale = (maxs[i] - mins[i]) / (s - 1)
        quantity = copy.deepcopy(sd.quantity[i])
        ad.add_dim(offset=offset, scale=scale, quantity=quantity)
    ad.add_dim(quantity=copy.deepcopy(sd.quantity[-1]))
    # Set sampling mesh
    mesh = np.indices(sampling_shape).astype(float)
    for i in range(dims):
        mesh[i] = ad.scale.offset[i] + ad.scale.scale[i] * mesh[i]
    mesh = mesh.reshape((dims, -1)).T
    # Interpolation
    data = interpolate.griddata(
        sd.data[:-1].T, sd.data[-1], mesh, method=algorithm, fill_value=fill
    )
    ad.data = data.reshape(sampling_shape)
    return ad


###############################################################################


if __name__ == "__main__":

    # Test data
    offset = 0
    end = 3
    num = 1001
    scale = (end - offset) / (num - 1)
    ar = np.cos(2 * np.pi * np.linspace(offset, end, num=num))
    ad = arraydata.ArrayData()
    ad.add_dim(offset=offset, scale=scale, name="time", symbol="t", unit="s")
    ad.add_dim(name="amplitude", symbol="A", unit="V")
    ad.data = ar
    # Convert to series data
    sd = cv_arraydata_to_seriesdata(ad)
    # Convert back to array data
    ad2 = cv_seriesdata_to_arraydata(sd)
    # Plot
    from libics.display import plot, plotdefault
    pcfg = [
        plotdefault.get_plotcfg_arraydata_1d(label="original"),
        plotdefault.get_plotcfg_seriesdata_1d(label="seriesdata"),
        plotdefault.get_plotcfg_arraydata_1d(label="arraydata")
    ]
    fcfg = plotdefault.get_figurecfg()
    fig = plot.Figure(fcfg, pcfg, data=[ad, sd, ad2])
    fig.plot()
    fig.legend()
    fig.show()
