# System Imports
import abc
import threading

# Package Imports
from libics import cfg
from libics import drv
from libics.util import InheritMap


###############################################################################


class DrvBase(abc.ABC):

    """
    Driver base class.

    Provides an API to communicate with the external interface. Communication
    is serialized with a message queue that can be processed. Immediate
    thread-safe communication is enabled by acquiring the `interface_access`
    lock and calling the `read`/`write` methods directly.

    The initialization functions (`setup`, `shutdown`, `connect`, `close`) have
    to be implemented as well as the actual functional methods. Methods using
    the message queue system have to implement I/O functions for each attribute
    and should be named `_write_<attr_name>` and `_read_<attr_name>`. The
    write function takes one value parameter and the read function returns a
    corresponding value parameter. When implementing direct access methods,
    guard the method with lock access.

    Parameters
    ----------
    cfg : DrvBaseCfg
        Driver configuration object.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._interface = None
        self.interface_access = threading.Lock()

    def __enter__(self):
        try:
            self.setup()
        except Exception:
            try:
                self.shutdown()
            except Exception:
                pass
            raise
        try:
            self.connect()
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise
        return self

    def __exit__(self, *args):
        self.close()
        self.shutdown()

    def setup(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg
        self._interface = drv.itf.get_itf(self.cfg.interface)
        self._interface.setup()

    def shutdown(self):
        self._interface.shutdown()

    def connect(self):
        self._interface.connect()

    def close(self):
        self._interface.close()

    def write(self, msg):
        func = getattr(self, "_write_" + msg.name)
        self.interface_access.acquire()
        if msg.value is None:
            func()
        else:
            func(msg.value)
        self.interface_access.release()

    def read(self, msg):
        func = getattr(self, "_read_" + msg.name)
        self.interface_access.acquire()
        ret = func()
        self.interface_access.release()
        msg.callback(ret)

    def process(self):
        """
        Processes the message queue in the configuration object.
        """
        msg = self.cfg._pop_msg()
        while (msg is not None):
            if (msg.msg_type == cfg.CFG_MSG_TYPE.WRITE or
                    msg.msg_type == cfg.CFG_MSG_TYPE.VALIDATE):
                self.write(msg)
            if (msg.msg_type == cfg.CFG_MSG_TYPE.READ or
                    msg.msg_type == cfg.CFG_MSG_TYPE.VALIDATE):
                self.read(msg)
            msg = self.cfg._pop_msg()

    def read_all(self):
        """
        Reads all configuration items from interface.
        Automatically processes the message queue.
        """
        self.cfg.read_all()
        self.process()

    def get_drv(self, cfg=None):
        if cfg is None:
            cfg = self.cfg
        return drv.get_drv(cfg)


###############################################################################


class DRV_DRIVER:

    CAM = 0         # Camera
    PIEZO = 10      # Piezo controller
    SPAN = 20       # Spectrum analyzer
    OSC = 30        # Oscilloscope
    DSP = 40        # Display


class DRV_MODEL:

    # Cam
    ALLIEDVISION_MANTA_G145B_NIR = 101

    # Piezo
    THORLABS_MDT69XA = 1101
    THORLABS_MDT693A = THORLABS_MDT69XA
    THORLABS_MDT694A = THORLABS_MDT69XA

    # SpAn
    STANFORD_SR760 = 2101
    YOKAGAWA_AQ6315 = 2201
    AGILENT_N9320X = 2301

    # Osc
    TEKTRONIX_TDS100X = 3101

    # Dsp
    TEXASINSTRUMENTS_DLP7000 = 4101


@InheritMap(map_key=("libics", "DrvCfgBase"))
class DrvCfgBase(cfg.CfgBase):

    """
    DrvCfgBase.

    Parameters
    ----------
    driver : DRV_DRIVER
        Driver type.
    interface : drv.itf.itf.ProtocolCfgBase
        Connection interface configuration.
    identifier : str
        Unique identifier of device.
    model : DRV_MODEL
        Device model.
    """

    driver = cfg.CfgItemDesc()
    identifier = cfg.CfgItemDesc()
    model = cfg.CfgItemDesc()

    def __init__(
        self,
        driver=DRV_DRIVER.CAM, interface=None, identifier="", model="",
        cls_name="DrvCfgBase", **kwargs
    ):
        super().__init__(cls_name=cls_name)
        self.driver = driver
        self.identifier = identifier
        self.model = model
        self._kwargs = kwargs
        if isinstance(interface, dict):
            self.interface = (
                drv.itf.itf.ProtocolCfgBase(**interface).get_hl_cfg()
            )
        else:
            self.interface = interface

    def get_hl_cfg(self):
        MAP = {
            DRV_DRIVER.CAM: CamCfg,
            DRV_DRIVER.PIEZO: PiezoCfg,
            DRV_DRIVER.SPAN: SpAnCfg,
            DRV_DRIVER.OSC: OscCfg,
            DRV_DRIVER.DSP: DspCfg
        }
        obj = MAP[self.driver.val](ll_obj=self, **self._kwargs)
        return obj.get_hl_cfg()

    def read_all(self):
        """
        Update all configuration items by reading all values from interface.
        """
        for key, val in self.__dict__.items():
            if isinstance(val, cfg.CfgItem):
                val.read()


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


@InheritMap(map_key=("libics", "CamCfg"))
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

    pixel_hrzt_count = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_hrzt_size = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_hrzt_offset = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_vert_count = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_vert_size = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_vert_offset = cfg.CfgItemDesc(group="format", val_check=(0, None))
    format_color = cfg.CfgItemDesc(group="format")
    channel_bitdepth = cfg.CfgItemDesc(group="format", val_check=int)
    exposure_mode = cfg.CfgItemDesc(group="capture")
    exposure_time = cfg.CfgItemDesc(group="capture", val_check=(0, None))
    acquisition_frames = cfg.CfgItemDesc(group="capture", val_check=int)
    sensitivity = cfg.CfgItemDesc(group="capture")

    def __init__(
        self,
        pixel_hrzt_count=1338, pixel_hrzt_size=6.45e-6,
        pixel_vert_count=1038, pixel_vert_size=6.45e-6,
        pixel_hrzt_offset=0, pixel_vert_offset=0,
        format_color=DRV_CAM.FORMAT_COLOR.GS, channel_bitdepth=8,
        exposure_mode=DRV_CAM.EXPOSURE_MODE.MANUAL, exposure_time=1e-3,
        acquisition_frames=0, sensitivity=DRV_CAM.SENSITIVITY.NORMAL,
        cls_name="CamCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.CAM
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
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


@InheritMap(map_key=("libics", "PiezoCfg"))
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

    limit_min = cfg.CfgItemDesc(group="limit")
    limit_max = cfg.CfgItemDesc(group="limit")
    displacement = cfg.CfgItemDesc(group="property")
    channel = cfg.CfgItemDesc()
    feedback_mode = cfg.CfgItemDesc(group="property")

    def __init__(
        self,
        limit_min=0.0, limit_max=75.0,
        displacement=2.67e-7,
        channel=None,
        feedback_mode=DRV_PIEZO.FEEDBACK_MODE.OPEN_LOOP,
        cls_name="PiezoCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.PIEZO
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
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


@InheritMap(map_key=("libics", "SpAnCfg"))
class SpAnCfg(DrvCfgBase):

    """
    DrvCfgBase -> SpAnCfg.

    Parameters
    ----------
    bandwidth : float
        Spectral bandwidth in Hertz (Hz).
    frequency_start, frequency_stop : float
        Frequency range (start, stop) in Hertz (Hz).
    average_mode : DRV_SPAN.AVERAGE_MODE
        Averaging mode.
    average_count : int
        Number of averages.
    voltage_max : float
        Voltage input max range in decibel volts (dBV).
    """

    bandwidth = cfg.CfgItemDesc(group="frequency", val_check=(0, None))
    frequency_start = cfg.CfgItemDesc(group="frequency")
    frequency_stop = cfg.CfgItemDesc(group="frequency")
    average_mode = cfg.CfgItemDesc(group="average")
    average_count = cfg.CfgItemDesc(group="average", val_check=(0, None))
    voltage_max = cfg.CfgItemDesc(group="amplitude")

    def __init__(
        self,
        bandwidth=1e3,
        frequency_start=0.0, frequency_stop=1e5,
        average_mode=DRV_SPAN.AVERAGE_MODE.LIN, average_count=100,
        voltage_max=-30.0,
        cls_name="SpAnCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.SPAN
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
        self.bandwidth = bandwidth
        self.frequency_start = frequency_start
        self.frequency_stop = frequency_stop
        self.average_mode = average_mode
        self.average_count = average_count
        self.voltage_max = voltage_max

    def get_hl_cfg(self):
        return self


@InheritMap(map_key=("libics", "OscCfg"))
class OscCfg(DrvCfgBase):

    """
    DrvCfgBase -> OscCfg.

    Parameters
    ----------
    """

    def __init__(
        self,
        cls_name="OscCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.OSC
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)

    def get_hl_cfg(self):
        return self


class DRV_DSP:

    class FORMAT_COLOR:

        BW = 0
        GS = 1
        RGB = 2
        RGBA = 3


@InheritMap(map_key=("libics", "DspCfg"))
class DspCfg(DrvCfgBase):

    """
    DrvCfgBase -> DspCfg.

    Parameters
    ----------
    pixel_hrzt_count, pixel_vert_count : int
        Pixel count in respective direction.
    pixel_hrzt_size, pixel_vert_size : float
        Pixel size in meters in respective direction.
    pixel_hrzt_offset, pixel_vert_offset : int
        Offset of pixels to be captured.
    format_color : DRV_DSP.FORMAT_COLOR
        BW: black/white boolean image.
        GS: greyscale image.
        RGB: RGB color image.
        RGBA: RGB image with alpha channel.
    channel_bitdepth : int
        Bits per color channel.
    picture_time : float
        Time in seconds (s) each single image is shown.
    dark_time : float
        Time in seconds (s) between images in a sequence.
    sequence_repetitions : int
        Number of sequences to be shown.
        0 (zero) is interpreted as infinite, i.e.
        continuos repetition.
    temperature : float
        Display temperature in Celsius (°C).
    """

    pixel_hrzt_count = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_hrzt_size = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_hrzt_offset = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_vert_count = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_vert_size = cfg.CfgItemDesc(group="format", val_check=(0, None))
    pixel_vert_offset = cfg.CfgItemDesc(group="format", val_check=(0, None))
    format_color = cfg.CfgItemDesc(group="format")
    channel_bitdepth = cfg.CfgItemDesc(group="format", val_check=int)
    picture_time = cfg.CfgItemDesc(group="dmd", val_check=(0, 9))
    dark_time = cfg.CfgItemDesc(group="dmd", val_check=(0, None))
    sequence_repetitions = cfg.CfgItemDesc(group="dmd", val_check=(0, None))
    temperature = cfg.CfgItemDesc(group="dmd")

    def __init__(
        self,
        pixel_hrzt_count=1024, pixel_hrzt_size=13.68e-6,
        pixel_vert_count=768, pixel_vert_size=13.68e-6,
        pixel_hrzt_offset=0, pixel_vert_offset=0,
        format_color=DRV_DSP.FORMAT_COLOR.GS, channel_bitdepth=8,
        picture_time=9.0, dark_time=0.0, sequence_repetitions=0,
        temperature=25.0,
        cls_name="DspCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.DSP
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
        self.pixel_hrzt_count = pixel_hrzt_count
        self.pixel_hrzt_size = pixel_hrzt_size
        self.pixel_hrzt_offset = pixel_hrzt_offset
        self.pixel_vert_count = pixel_vert_count
        self.pixel_vert_size = pixel_vert_size
        self.pixel_vert_offset = pixel_vert_offset
        self.format_color = format_color
        self.channel_bitdepth = channel_bitdepth
        self.picture_time = picture_time
        self.dark_time = dark_time
        self.sequence_repetitions = sequence_repetitions
        self.temperature = temperature

    def get_hl_cfg(self):
        return self
