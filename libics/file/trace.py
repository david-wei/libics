import numpy as np

from libics.data import arraydata, dataset
from libics.drv import drv
from libics.file import traceutil


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
    AttributeError
        If the wct file is corrupt.
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
