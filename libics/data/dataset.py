from libics.data import stp
from libics.drv import drv
from libics.file import hdf
from libics.util import InheritMap


###############################################################################


@InheritMap(map_key=("libics", "DataSet"))
class DataSet(hdf.HDFBase):

    """
    Container class wrapping actual data with configuration/setup data.

    Parameters
    ----------
    data : arraydata.ArrayData or seriesdata.SeriesData
        Storage for actual data.
    cfg : drv.drv.DrvCfgBase
        Storage for driver configuration.
    stp : stp.SetupCfgBase
        Storage for (measurement) setup configuration.
    """

    def __init__(
        self,
        data=None,
        cfg=drv.DrvCfgBase(), stp=stp.SetupCfgBase()
    ):
        self.data = data
        self.cfg = cfg
        self.stp = stp
