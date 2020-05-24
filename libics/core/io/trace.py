import numbers

import numpy as np

from libics.data import arraydata, seriesdata, dataset, types
from libics.drv import drv
from libics.file import traceutil
from libics.util import misc


###############################################################################


def load_csv_span_agilent_to_dataset(file_path):
    """
    Reads an Agilent spectrum analyzer text (csv) file and loads the data
    into a `data.dataset.DataSet` structure with `data.arraydata.ArrayData`
    as underlying data storage.

    Parameters
    ----------
    file_path : `str`
        Path to the Agilent spectrum analyzer text file.

    Returns
    -------
    ds : `data.dataset.DataSet`
        Power spectral density.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    """
    # Parse data
    _, ar, md = traceutil.parse_csv_span_agilent_to_numpy_array(file_path)
    cfg = {}
    cfg["driver"] = drv.DRV_DRIVER.SPAN
    cfg["interface"] = None
    cfg["model"] = None
    if "N9320" in md["Model"].val:
        cfg["model"] = drv.DRV_MODEL.AGILENT_N9320X
    cfg["identifier"] = cfg["model"]
    if "Serial Number" in md.keys():
        cfg["identifier"] = md["Serial Number"].val
    bandwidth = float(md["Resolution Bandwidth"].val)
    cfg["bandwidth"] = bandwidth
    freq_center = float(md["Center Frequency"].val)
    freq_span = float(md["Span"].val)
    cfg["frequency_start"] = freq_center - freq_span / 2
    cfg["frequency_stop"] = freq_center + freq_span / 2
    # Convert from dBm via 50 Ohms to dBV
    vmax_dbm = float(md["Reference Level"].val)
    cfg["voltage_max"] = vmax_dbm + 10 * np.log10(50 / 2**2) - 30
    # Convert spectrum to spectral density (dBm -> dBV/√Hz)
    num = int(md["Num Points"].val) - 1
    ar = ar + 10 * np.log10(50 / 2**2) - 30 - 10 * np.log10(bandwidth)
    # Setup variables
    cfg = drv.SpAnCfg(**cfg)
    ad = arraydata.ArrayData()
    ad.add_dim(
        offset=cfg.frequency_start.val,
        scale=((cfg.frequency_stop.val - cfg.frequency_start.val) / num),
        quantity=md["_frequency_unit"]
    )
    ad.add_dim(
        name="power spectral density", symbol="PSD", unit="dBV/√Hz"
    )
    ad.data = ar
    ds = dataset.DataSet(data=ad, cfg=cfg)
    return ds


def load_txt_span_numpy_to_dataset(
    file_path, spectral_quantity=None, weight_quantity=None,
    normalization=None
):
    """
    Reads a numpy spectrum text (txt) file and loads the data into a
    `data.dataset.DataSet` structure with `data.seriesdata.SeriesData`
    as underlying data storage.
    Assumes a 2D array with spectral data in the first dimension and
    the spectral weight in the second dimension.

    Parameters
    ----------
    file_path : `str`
        Path to the Agilent spectrum analyzer text file.
    spectral_quantity : `data.types.Quantity`
        Quantity description for spectral data (usually
        frequency or wavelength).
    weight_quantity : `data.types.Quantity`
        Quantity description for spectral weight (usually
        relative or power density).
    normalization : None or str or float
        None:
            No normalization performed.
        "max":
            Normalizes to the weights' maximum.
        "sum":
            Normalizes to the weights' sum.
        "weighted_sum":
            Normalizes to the weighted sum, i.e.
            weighs the sum by the spectral bin size.
        float:
            Normalizes such that the maximum has the
            given value.

    Returns
    -------
    ds : `data.dataset.DataSet`
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
    sd = seriesdata.SeriesData()
    sd.add_dim(quantity=spectral_quantity)
    sd.add_dim(quantity=weight_quantity)
    sd.data = spectrum
    # Setup data set
    ds = dataset.DataSet(data=sd)
    return ds
