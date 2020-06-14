import queue
import time

from libics.core.env import logging
from libics.driver.device import STATUS
from libics.driver.interface import ItfBase
from libics.driver.camera import Camera
from libics.driver.camera import EXPOSURE_MODE, FORMAT_COLOR, SENSITIVITY

LOGGER = logging.get_logger("libics.driver.camera.vimba")

try:
    import pymba
except ImportError:
    LOGGER.info(
        "Could not load AlliedVision Vimba API. "
        + "If you are using a Vimba camera, install the Vimba C API and "
        + "the Python wrapper `pymba`."
    )


###############################################################################
# Interface
###############################################################################


class ItfVimba(ItfBase):

    # Vimba API hooks
    _vimba_itf = None
    _vimba_system = None
    # Referemces to Vimba API
    _vimba_itf_refs = 0
    # Vimba device references
    _vimba_dev_refs = {}
    # Vimba device handles
    _vimba_dev_handles = {}

    def __init__(self):
        super().__init__()
        self._dev_id = None

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    def setup(self):
        if not self.is_set_up():
            self._vimba_itf = pymba.vimba.Vimba()
            self._vimba_itf.startup()
            self._vimba_system = self._vimba_itf.getSystem()
        self._vimba_itf_refs += 1

    def shutdown(self):
        if self.is_set_up():
            self._vimba_itf_refs -= 1
            if self._vimba_itf_refs == 0:
                self._vimba_itf.shutdown()
                self._vimba_itf = None
                self._vimba_system = None

    def is_set_up(self):
        return self._vimba_itf_refs > 0

    def connect(self, id):
        """
        Raises
        ------
        RuntimeError
            If `id` is not available.
        """
        if self.is_connected():
            if self._dev_id == id:
                return
            else:
                self.close()
        # Check if requested device ID is discovered
        if id not in self._vimba_dev_refs:
            self.discover()
            if id not in self._vimba_dev_refs:
                raise RuntimeError("device ID unavailable ({:s})".format(id))
        # Set device ID of this interface instance
        self._dev_id = id
        # Check if another interface instance has already opened
        if not self.is_connected():
            dev_handle = self._vimba_itf.getCamera(self._dev_id)
            self._vimba_dev_handles[self._dev_id] = dev_handle
        self._vimba_dev_refs[self._dev_id] += 1

    def close(self):
        self._vimba_dev_refs[self._dev_id] -= 1
        if self._vimba_dev_refs[self._dev_id] == 0:
            del self._vimba_dev_handles[self._dev_id]

    def is_connected(self):
        """
        Raises
        ------
        RuntimeError
            If internal device reference error occured.
        """
        if self._dev_id is None:
            return False
        try:
            # If not discovered
            if self._dev_id not in self._vimba_dev_refs:
                assert(self._dev_id not in self._vimba_dev_handles)
                return False
            # If discovered but not connected
            elif self._vimba_dev_refs[self._dev_id] == 0:
                assert(self._dev_id not in self._vimba_dev_handles)
                return False
            # If discovered and connected
            elif self._vimba_dev_refs[self._dev_id] > 0:
                assert(self._dev_id in self._vimba_dev_handles)
                return True
            # Handle error
            else:
                assert(False)
        except AssertionError:
            err_msg = "device reference count error"
            self.last_status = STATUS(
                state=STATUS.CRITICAL, err_type=STATUS.ERR_INSTANCE,
                msg=err_msg
            )
            raise RuntimeError(err_msg)

    # ++++++++++++++++++++++++++++++++++++++++
    # Interface methods
    # ++++++++++++++++++++++++++++++++++++++++

    @classmethod
    def discover(cls):
        TIMEOUT_DISCOVERY_GIGE = 0.2
        if cls._vimba_system.GeVTLIsPresent:
            cls._vimba_system.runFeatureCommand("GeVDiscoveryAllOnce")
            time.sleep(TIMEOUT_DISCOVERY_GIGE)
        dev_ids = cls._vimba_itf.getCameraIds()
        # Check for devices which have become unavailable
        for id in list(cls._vimba_dev_refs):
            if id not in dev_ids:
                del cls._vimba_dev_refs[id]
                if id in cls._vimba_dev_handles:
                    cls.LOGGER.critical(
                        "device lost connection ({:s})".format(id)
                    )
                    del cls._vimba_dev_handles[id]
                    # TODO: notify affected devices
        # Check for devices which have been added
        for id in dev_ids:
            if id not in cls._vimba_dev_refs:
                cls._vimba_dev_refs[id] = 0
        return dev_ids


###############################################################################
# Device
###############################################################################


class AlliedVisionManta(Camera):

    def __init__(self):
        super().__init__()
        self.properties.set_properties(self._get_default_properties_dict(
            "device_name", "sensitivity"
        ))
        self._dev_frame = None

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    def setup(self):
        if not isinstance(self.interface, ItfVimba):
            self.interface = ItfVimba()
        self.interface.setup()

    def shutdown(self):
        self.interface.shutdown()

    def is_set_up(self):
        return self.interface.is_set_up()

    def connect(self):
        self.interface.connect(self.identifier)
        self.interface.register(self.identifier, self)
        self.vimba_dev_handle.openCamera(cameraAccessMode=0)
        self.p.read_all()

    def close(self):
        self.vimba_dev_handle.closeCamera()
        self.interface.deregister(id=self.identifier)
        self.interface.close()

    def is_connected(self):
        return self.interface.is_connected()

    # ++++++++++++++++++++++++++++++++++++++++
    # Camera methods
    # ++++++++++++++++++++++++++++++++++++++++

    def _start_acquisition(self):
        self._dev_frame = self.vimba_dev_handle.getFrame()
        self._dev_frame.announceFrame()
        self.vimba_dev_handle.startCapture()
        self._dev_frame.queueFrameCapture()
        self.vimba_dev_handle.runFeatureCommand("AcquisitionStart")

    def _end_acquisition(self):
        self.vimba_dev_handle.runFeatureCommand("AcquisitionStop")
        self.vimba_dev_handle.flushCaptureQueue()
        self.vimba_dev_handle.endCapture()
        self._dev_frame.revokeFrame()
        self._dev_frame = None

    def next(self):
        TIMEOUT_FRAME = int(self.p.exposure_time * 1200)   # milliseconds
        if self._dev_frame.waitFrameCapture(timeout=TIMEOUT_FRAME) == 0:
            np_image = self._cv_buffer_to_numpy(
                self._dev_frame.getBufferByteData()
            )
            self._dev_frame.queueFrameCapture()
            # Save to buffer
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(np_image)
            return np_image
        else:
            return self.next()

    # ++++++++++++++++++++++++++++++++++++++++
    # Helper methods
    # ++++++++++++++++++++++++++++++++++++++++

    @property
    def vimba_dev_handle(self):
        return self.interface._vimba_dev_handles[self.identifier]

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_pixel_hrzt_count(self):
        value = self.vimba_dev_handle.Width
        self.p.pixel_hrzt_count = value
        return value

    def write_pixel_hrzt_count(self, value):
        self.vimba_dev_handle.Width = value
        self.p.pixel_hrzt_count = value

    def read_pixel_hrzt_size(self):
        value = 6.45e-6
        self.p.pixel_hrzt_size = value
        return value

    def write_pixel_hrzt_size(self, value):
        if value != self.p.pixel_hrzt_size:
            self.LOGGER.warning("cannot write pixel_hrzt_size")

    def read_pixel_hrzt_offset(self):
        value = self.vimba_dev_handle.OffsetX
        self.p.pixel_hrzt_offset = value
        return value

    def write_pixel_hrzt_offset(self, value):
        self.vimba_dev_handle.OffsetX = value
        self.p.pixel_hrzt_offset = value

    def read_pixel_vert_count(self):
        value = self.vimba_dev_handle.Height
        self.p.pixel_vert_count = value
        return value

    def write_pixel_vert_count(self, value):
        self.vimba_dev_handle.Height = value
        self.p.pixel_vert_count = value

    def read_pixel_vert_size(self):
        value = 6.45e-6
        self.p.pixel_vert_size = value
        return value

    def write_pixel_vert_size(self, value):
        if value != self.p.pixel_vert_size:
            self.LOGGER.warning("cannot write pixel_vert_size")

    def read_pixel_vert_offset(self):
        value = self.vimba_dev_handle.OffsetY
        self.p.pixel_vert_offset = value
        return value

    def write_pixel_vert_offset(self, value):
        self.vimba_dev_handle.OffsetY = value
        self.p.pixel_vert_offset = value

    def read_format_color(self):
        value = FORMAT_COLOR.GS
        self.format_color = value
        return value

    def write_format_color(self, value):
        if value != self.p.format_color:
            self.LOGGER.warning("cannot write format_color")

    def read_channel_bitdepth(self):
        MAP = {"Mono8": 8, "Mono12": 12, "Mono12Packed": 12}
        value = MAP[self.vimba_dev_handle.PixelFormat]
        self.channel_bitdepth = value
        return value

    def write_channel_bitdepth(self, value):
        MAP = {8: "Mono8", 12: "Mono12"}
        self.vimba_dev_handle.PixelFormat = MAP[value]
        self.p.channel_bitdepth = value

    def read_exposure_mode(self):
        MAP = {
            "Off": EXPOSURE_MODE.MANUAL,
            "Continuous": EXPOSURE_MODE.CONTINUOS,
            "Single": EXPOSURE_MODE.SINGLE
        }
        value = MAP[self.vimba_dev_handle.ExposureAuto]
        self.p.exposure_mode = value
        return value

    def write_exposure_mode(self, value):
        MAP = {
            EXPOSURE_MODE.MANUAL: "Off",
            EXPOSURE_MODE.CONTINUOS: "Continuous",
            EXPOSURE_MODE.SINGLE: "Single"
        }
        self.vimba_dev_handle.ExposureAuto = MAP[value]
        self.p.exposure_mode = value

    def read_exposure_time(self):
        value = self.vimba_dev_handle.ExposureTimeAbs / 1e6
        self.p.exposure_time = value
        return value

    def write_exposure_time(self, value):
        self.vimba_dev_handle.ExposureTimeAbs = value * 1e6
        self.p.exposure_time = value

    def read_acquisition_frames(self):
        value = self.vimba_dev_handle.AcquisitionMode
        MAP = {
            "Continuous": 0,
            "SingleFrame": 1,
            "MultiFrame": self._interface.cam.AcquisitionFrameCount
        }
        value = MAP[value]
        self.p.acquisition_frames = value
        return value

    def write_acquisition_frames(self, value):
        if value == 0:
            self._interface.cam.AcquisitionMode = "Continuous"
        elif value == 1:
            self._interface.cam.AcquisitionMode = "SingleFrame"
        else:
            self._interface.cam.AcquisitionMode = "MultiFrame"
            self._interface.cam.AcquisitionFrameCount = value
        self.p.acquisition_frames = value

    def read_device_name(self):
        name = self.vimba_dev_handle.getInfo().cameraName.decode("ascii")
        self.p.device_name = name
        return name

    def write_device_name(self, value):
        if value != self.p.device_name:
            self.LOGGER.warning("cannot write device_name")

    def read_sensitivity(self):
        MAP = {
            "Off": SENSITIVITY.NORMAL,
            "On_Fast": SENSITIVITY.NIR_FAST,
            "On_HighQuality": SENSITIVITY.NIR_HQ
        }
        value = MAP[self.vimba_dev_handle.NirMode]
        self.p.sensitivity = value
        return value

    def write_sensitivity(self, value):
        MAP = {
            SENSITIVITY.NORMAL: "Off",
            SENSITIVITY.NIR_FAST: "On_Fast",
            SENSITIVITY.NIR_HQ: "On_HighQuality",
        }
        self.vimba_dev_handle.NirMode = MAP[value]
        self.p.sensitivity = value
