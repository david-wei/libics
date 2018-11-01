from . import itf           # noqa
from . import cam           # noqa
from . import camutil       # noqa
from . import drv           # noqa
from . import drvcam        # noqa
from . import drvosc        # noqa
from . import drvpiezo      # noqa
from . import drvspan       # noqa
from . import piezo         # noqa


###############################################################################


def get_drv(cfg):
    """
    Gets the driver object from the given driver configuration.

    Parameters
    ----------
    cfg : drv.DrvCfgBase
        Driver configuration object.

    Returns
    -------
    drv : drv.DrvBase
        Requested driver object.
    """
    if cfg.driver == drv.DRV_DRIVER.CAM:
        return drvcam.get_cam_drv(cfg)
    elif cfg.driver == drv.DRV_DRIVER.PIEZO:
        return drvpiezo.get_piezo_drv(cfg)
    elif cfg.driver == drv.DRV_DRIVER.SPAN:
        return drvspan.get_span_drv(cfg)
    elif cfg.driver == drv.DRV_DRIVER.OSC:
        return drvosc.get_osc_drv(cfg)
