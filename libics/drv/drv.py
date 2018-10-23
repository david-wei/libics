# Package Imports
from libics import cfg


###############################################################################


class DRV_DRIVER:

    CAM = 0
    PIEZO = 10


class DrvCfgBase(cfg.CfgBase):

    """
    DrvCfgBase.

    Parameters
    ----------
    driver : DRV_DRIVER
        Driver type.
    interface : itf.itf.ProtocolCfgBase
        Connection interface configuration.
    identifier : str
        Unique identifier of device.
    """

    def __init__(self, driver=DRV_DRIVER.CAM, interface=None, identifier=""):
        self.driver = driver
        self.interface = interface
        self.identifier = identifier

    def get_hl_cfg(self):
        obj = DRV_DRIVER.MAP[self.interface](ll_obj=self, **self.kwargs)
        return obj.get_hl_cfg()


###############################################################################


class DRV_CAM:

    class FORMAT_COLOR:

        BW = 0
        GS = 1
        RGB = 2
        RGBA = 3

    class EXPOSURE_MODE:

        MANUAL = 0
        CONTINUOS = 1
        SINGLE = 2

    class SENSITIVITY:

        NORMAL = 0
        NIR_FAST = 1
        NIR_HQ = 2


class CamCfg(DrvCfgBase):

    """
    DrvCfgBase -> CamCfg.

    Parameters
    ----------
    pixel_hrzt_count, pixel_vert_count : int
        Pixel count in respective direction.
    pixel_hrzt_size, pixel_vert_size : float
        Pixel size in meters in respective direction.
    pixel_hrzt_offset, pixel_vert_offset : int
        Offset of pixels to be captured.
    format_color : DRV_CAM.FORMAT_COLOR
        BW: black/white boolean image.
        GS: greyscale image.
        RGB: RGB color image.
        RGBA: RGB image with alpha channel.
    channel_bitdepth : int
        Bits per color channel.
    exposure_mode : DRV_CAM.EXPOSURE_MODE
        MANUAL: manual, fixed exposure time.
        CONTINUOS: continuosly self-adjusted exposure time.
        SINGLE: fixed exposure time after single adjustment.
    exposure_time : float
        Exposure time in seconds.
    acquisition_frames : int
        Number of frames to be acquired.
        0 (zero) is interpreted as infinite, i.e.
        continuos acquisition.
    sensitivity : DRV_CAM.SENSITIVITY
        NORMAL: normal acquisition.
        NIR_FAST: fast near-IR enhancement.
        NIR_HQ: high-quality near-IR enhancement.

    Notes
    -----
    * Horizontal (hrzt) and vertical (vert) directions.
    """

    def __init__(
        self,
        pixel_hrzt_count=1338, pixel_hrzt_size=6.45e-6,
        pixel_vert_count=1038, pixel_vert_size=6.45e-6,
        pixel_hrzt_offset=0, pixel_vert_offset=0,
        format_color=DRV_CAM.FORMAT_COLOR.GS, channel_bitdepth=8,
        exposure_mode=DRV_CAM.EXPOSURE_MODE.MANUAL, exposure_time=1e-3,
        acquisition_frames=0, sensitivity=DRV_CAM.SENSITIVITY.NORMAL
    ):
        self.pixel_hrzt_count = pixel_hrzt_count
        self.pixel_hrzt_size = pixel_hrzt_size
        self.pixel_hrzt_offset = pixel_hrzt_offset
        self.pixel_vert_count = pixel_vert_count
        self.pixel_vert_size = pixel_vert_size
        self.pixel_vert_offset = pixel_vert_offset
        self.format_color = format_color
        self.channel_bitdepth = channel_bitdepth
        self.exposure_mode = exposure_mode
        self.exposure_time = exposure_time
        self.acquisition_frames = acquisition_frames
        self.sensitivity = sensitivity

    def get_hl_cfg(self):
        return self


class PiezoCfg(DrvCfgBase):

    """
    DrvCfgBase -> PiezoCfg.

    Parameters
    ----------
    limit_min, limit_max : float
        Voltage limit minimum and maximum in volts.
    displacement : float
        Displacement in meters per volt.
    """

    def __init__(
        self,
        limit_min=0.0, limit_max=75.0,
        displacement=2.67e-7
    ):
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.displacement = displacement

    def get_hl_cfg(self):
        return self


# +++++++++++++++++++++++++++++


DRV_DRIVER.MAP = {
    DRV_DRIVER.CAM: CamCfg,
    DRV_DRIVER.PIEZO: PiezoCfg
}
