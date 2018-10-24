# Package Imports
from libics import cfg


###############################################################################


class DRV_DRIVER:

    CAM = 0         # Camera
    PIEZO = 10      # Piezo controller
    SPAN = 20       # Spectrum analyzer
    OSC = 30        # Oscilloscope


class DRV_MODEL:

    # Cam
    ALLIEDVISION_MANTA_G145B_NIR = 101

    # Piezo
    THORLABS_MDT69XA = 1101
    THORLABS_MDT693A = THORLABS_MDT69XA
    THORLABS_MDT694A = THORLABS_MDT69XA

    # SpAn
    STANFORD_SR760 = 2101

    # Osc
    TEKTRONIX_TDS100X = 3101


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
    model : str
        Device model.
    """

    def __init__(
        self,
        driver=DRV_DRIVER.CAM, interface=None, identifier="", model=""
    ):
        super().__init__()
        self.driver = driver
        self.interface = interface
        self.identifier = identifier
        self.model = model

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
        Exposure time in seconds (s).
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
        acquisition_frames=0, sensitivity=DRV_CAM.SENSITIVITY.NORMAL,
        ll_obj=None, **kwargs
    ):
        super().__init__(**kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
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


class DRV_PIEZO:

    class FEEDBACK_MODE:

        OPEN_LOOP = 0
        CLOSED_LOOP = 1


class PiezoCfg(DrvCfgBase):

    """
    DrvCfgBase -> PiezoCfg.

    Parameters
    ----------
    limit_min, limit_max : float
        Voltage limit minimum and maximum in volts (V).
    displacement : float
        Displacement in meters per volt (m/V).
    channel : int or str
        Voltage channel.
    feedback_mode : DRV_PIEZO.FEEDBACK_MODE
        Feedback operation mode.
    """

    def __init__(
        self,
        limit_min=0.0, limit_max=75.0,
        displacement=2.67e-7,
        channel=None,
        feedback_mode=DRV_PIEZO.FEEDBACK_MODE.OPEN_LOOP,
        ll_obj=None, **kwargs
    ):
        super().__init__(**kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.displacement = displacement
        self.channel = channel
        self.feedback_mode = feedback_mode

    def get_hl_cfg(self):
        return self


class DRV_SPAN:

    class AVERAGE_MODE:

        LIN = 0
        EXP = 1


class SpAnCfg(DrvCfgBase):

    """
    DrvCfgBase -> SpAnCfg.

    Parameters
    ----------
    bandwith : float
        Spectral bandwidth in Hertz (Hz).
    frequency_start, frequency_stop : float
        Frequency range (start, stop) in Hertz (Hz).
    average_mode : DRV_SPAN.AVERAGE_MODE
        Averaging mode.
    average_count : int
        Number of averages.
    """

    def __init__(
        self,
        bandwidth=1e3,
        frequency_start=0.0, frequency_stop=1e5,
        average_mode=DRV_SPAN.AVERAGE_MODE.LIN, average_count=100,
        ll_obj=None, **kwargs
    ):
        super().__init__(**kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.bandwidth = bandwidth
        self.frequency_start = frequency_start
        self.frequency_stop = frequency_stop
        self.average_mode = average_mode
        self.average_count = average_count

    def get_hl_cfg(self):
        return self


class OscCfg(DrvCfgBase):

    """
    DrvCfgBase -> OscCfg.

    Parameters
    ----------
    """

    def __init__(
        self,
        ll_obj=None, **kwargs
    ):
        super().__init__(**kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)

    def get_hl_cfg(self):
        return self


# +++++++++++++++++++++++++++++


DRV_DRIVER.MAP = {
    DRV_DRIVER.CAM: CamCfg,
    DRV_DRIVER.PIEZO: PiezoCfg,
    DRV_DRIVER.SPAN: SpAnCfg,
    DRV_DRIVER.OSC: OscCfg
}
