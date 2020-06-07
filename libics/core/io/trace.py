import numbers
import numpy as np

from libics.core.data import types
from libics.core.data.arrays import ArrayData
from libics.core.io import traceutil
from libics.core.util import misc


###############################################################################


def load_csv_span_agilent_to_arraydata(file_path):
    """
    Reads an Agilent spectrum analyzer text (csv) file into an ArrayData.

    Parameters
    ----------
    file_path : `str`
        Path to the Agilent spectrum analyzer text file.

    Returns
    -------
    ds : `dict`
        Power spectral density data set containing metadata
        `"data"` : `data.arrays.ArrayData`
            Power spectral density.
        Further attributes: `"

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    """
    # Parse data
    _, ar, md = traceutil.parse_csv_span_agilent_to_numpy_array(file_path)
    ds = {}
    if "N9320" in md["Model"].val:
        ds["model"] = "AGILENT_N9320X"
    if "Serial Number" in md.keys():
        ds["identifier"] = md["Serial Number"].val
    bandwidth = float(md["Resolution Bandwidth"].val)
    ds["bandwidth"] = bandwidth
    freq_center = float(md["Center Frequency"].val)
    freq_span = float(md["Span"].val)
    ds["frequency_start"] = freq_center - freq_span / 2
    ds["frequency_stop"] = freq_center + freq_span / 2
    # Convert from dBm via 50 Ohms to dBV
    vmax_dbm = float(md["Reference Level"].val)
    ds["voltage_max"] = vmax_dbm + 10 * np.log10(50 / 2**2) - 30
    # Convert spectrum to spectral density (dBm -> dBV/√Hz)
    num = int(md["Num Points"].val) - 1
    ar = ar + 10 * np.log10(50 / 2**2) - 30 - 10 * np.log10(bandwidth)
    # Setup variables
    ad = ArrayData()
    ad.add_dim(1)
    ad.set_dim(
        0, offset=ds["frequency_start"].val,
        step=((ds["frequency_stop"].val - ds["frequency_start"].val) / num),
    )
    ad.set_var_quantity(0, quantity=md["_frequency_unit"])
    ad.set_data_quantity(
        name="power spectral density", symbol="PSD", unit="dBV/√Hz"
    )
    ad.data = ar
    ds["data"] = ad
    return ds


def load_txt_span_numpy_to_arraydata(
    file_path, spectral_quantity=None, weight_quantity=None,
    normalization=None
):
    """
    Reads a numpy spectrum text (txt) file into an `ArrayData`.

    Assumes a 2D array with spectral data in the first dimension and
    the spectral weight in the second dimension.

    Parameters
    ----------
    file_path : `str`
        Path to the numpy text file.
    spectral_quantity : `data.types.Quantity`
        Quantity description for spectral data (usually
        frequency or wavelength).
    weight_quantity : `data.types.Quantity`
        Quantity description for spectral weight (usually
        relative or power density).
    normalization : `None` or `str` or `float`
        `None`:
            No normalization performed.
        `"max"`:
            Normalizes to the weights' maximum.
        `"sum"`:
            Normalizes to the weights' sum.
        `"weighted_sum"`:
            Normalizes to the weighted sum, i.e.
            weighs the sum by the spectral bin size.
        `float`:
            Normalizes such that the maximum has the
            given value.

    Returns
    -------
    ad : `data.arrays.ArrayData`
        Spectral density.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    ValueError
        If quantities are invalid.
    """
    # Construct quantities
    misc.assume_construct_obj(
        spectral_quantity, types.Quantity, raise_exception=ValueError
    )
    if weight_quantity is None and normalization is not None:
        weight_quantity = types.Quantity(
            name="spectral density", symbol="S_A", unit="rel."
        )
    misc.assume_construct_obj(
        weight_quantity, types.Quantity, raise_exception=ValueError
    )
    # Load spectrum
    spectrum = np.loadtxt(file_path)
    if normalization is not None:
        spec, weights = spectrum
        if normalization == "max":
            weights /= weights.max()
        elif normalization == "sum":
            weights /= weights.sum()
        elif normalization == "weighted_sum":
            bin_sizes = abs(spec[1:] - spec[:-1])
            bin_sizes = np.array(
                [bin_sizes[0]]
                + list((bin_sizes[1:] + bin_sizes[:-1]) / 2)
                + [bin_sizes[-1]]
            )
            total_weight = np.sum(weights * bin_sizes)
            weights /= total_weight
        elif isinstance(normalization, numbers.Number):
            weights *= normalization / weights.max()
        spectrum = np.array([spec, weights])
    ad = ArrayData()
    ad.add_dim(quantity=spectral_quantity, points=spectrum[0])
    ad.data_quantity(quantity=weight_quantity)
    ad.data = spectrum[-1]
    return ad
