import copy

import numpy as np
from scipy import interpolate

from libics.data import arraydata, seriesdata, types
from libics.file import hdf
from libics.util import InheritMap, misc


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


@InheritMap(map_key=("libics", "Calibration"))
class Calibration(hdf.HDFBase):

    """
    Container class for storing calibration data mapping one quantity to
    another. Naming convention: key --map--> val.

    Parameters
    ----------
    key_data, val_data : np.ndarray(1)
        Data array for (key, val) data values.
    key_quantity, val_quantity : data.types.Quantity
        Quantity of (key, val).
    mode : Calibration.MODE
        NEAREST: Nearest neighbour.
        LINEAR: first order spline.
        QUADRATIC: second order spline.
        CUBIC: third order spline.
        PREVIOUS: previous value.
        NEXT: next value.
    extrapolation : bool or float or (float, float)
        True: Performs proper extrapolation.
        False: Uses boundary values.
        float: Uses given number as constant extrapolation.
        (float, float): Used as (left, right) extrapolation.
    """

    class MODE:

        NEAREST = "const"
        LINEAR = "linear"
        QUADRATIC = "quadratic"
        CUBIC = "cubic"
        PREVIOUS = "previous"
        NEXT = "next"

    def __init__(
        self, key_data, val_data, key_quantity, val_quantity,
        mode=MODE.LINEAR, extrapolation=False,
        pkg_name="libics", cls_name="Calibration"
    ):
        super().__init__(pkg_name=pkg_name, cls_name=cls_name)
        order = np.argsort(key_data)
        self.key_data = np.array(key_data)[order]
        self.val_data = np.array(val_data)[order]
        self.key_quantity = misc.assume_construct_obj(key_quantity,
                                                      types.Quantity)
        self.val_quantity = misc.assume_construct_obj(val_quantity,
                                                      types.Quantity)
        self.set_interpolation_mode(mode=mode, extrapolation=extrapolation)

    def set_interpolation_mode(self, mode=MODE.LINEAR, extrapolation=False):
        self.mode = mode
        self.extrapolation = extrapolation
        fill_value = extrapolation
        if self.extrapolation is True:
            fill_value = "extrapolate"
        elif self.extrapolation is False:
            fill_value = (self.val_data[0], self.val_data[-1])
        self.__interpolation = interpolate.interp1d(
            self.key_data, self.val_data, kind=self.mode, copy=False,
            assume_sorted=True, fill_value=fill_value
        )

    def __call__(self, *args):
        self.__interpolation(*args)

    def _hdf_init_write(self):
        # TODO: proper hdf implementation
        self.__interpolation = None


def apply_calibration(sd, calibration, dim=0):
    """
    Applies a calibration to a data.seriesdata.SeriesData object.

    Parameters
    ----------
    sd : data.seriesdata.SeriesData
        Series data the calibration is applied to.
    calibration : Calibration
        Calibration data.
    dim : int
        Series data dimension to which calibration is applied.

    Returns
    -------
    sd : data.seriesdata.SeriesData
        Series data with applied calibration.

    Notes
    -----
    * Performs in-place calibration, i.e. sd is mutable.
    * Does not check for quantity agreement. After applying calibration,
      sd quantity is changed to the quantity stored in calibration.
    """
    sd.data[dim] = calibration(sd.data[dim])
    sd.quantity[dim] = calibration.val_quantity
    return sd


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
