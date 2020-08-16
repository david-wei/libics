from . import vrmusbcam2 as vrm

import ctypes as ct
import numpy as np
import queue

from libics.env import logging
from libics.driver.device import STATUS
from libics.driver.interface import ItfBase
from libics.driver.camera import Camera, EXPOSURE_MODE, FORMAT_COLOR


###############################################################################
# Interface
###############################################################################


class ItfVRmagic(ItfBase):

    # vrm device reference counter
    _vrm_dev_refs = {}      # dev_id (str) -> refs (int)
    # vrm device key
    _vrm_dev_keys = {}      # dev_id (str) -> vrm_dev_key (ct)
    # vrm device handle
    _vrm_dev_handles = {}   # dev_id (str) -> vrm_dev_handle (ct)

    LOGGER = logging.get_logger("libics.driver.camera.vrmagic.ItfVRmagic")

    def __init__(self):
        super().__init__()
        self._is_set_up = False
        self._dev_id = None

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    def setup(self):
        self._is_set_up = True

    def shutdown(self):
        self._is_set_up = False

    def is_set_up(self):
        return self._is_set_up

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
        if id not in self._vrm_dev_refs:
            self.discover()
            if id not in self._vrm_dev_refs:
                raise RuntimeError(
                    "device ID unavailable ({:s})".format(str(id))
                )
        # Set device ID of this interface instance
        self._dev_id = id
        # Check if another interface instance has already opened
        if not self.is_connected():
            dev_keys = self._get_vrm_dev_key_list()
            for dev_key in dev_keys:
                if dev_key.contents.m_serial == self._dev_id:
                    self._vrm_dev_keys[self._dev_id] = dev_key
                else:
                    vrm.VRmUsbCamFreeDeviceKey(dev_key)
            dev_handle = vrm.VRmUsbCamDevice()
            vrm.VRmUsbCamOpenDevice(
                self._vrm_dev_keys[self._dev_id], ct.byref(dev_handle)
            )
            self._vrm_dev_handles[self._dev_id] = dev_handle
        self._vrm_dev_refs[self._dev_id] += 1

    def close(self):
        self._vrm_dev_refs[self._dev_id] -= 1
        if self._vrm_dev_refs[self._dev_id] == 0:
            vrm.VRmUsbCamCloseDevice(self._vrm_dev_handles[self._dev_id])
            del self._vrm_dev_handles[self._dev_id]
            vrm.VRmUsbCamFreeDeviceKey(self._vrm_dev_keys[self._dev_id])
            del self._vrm_dev_keys[self._dev_id]

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
            if self._dev_id not in self._vrm_dev_refs:
                assert(
                    (self._dev_id not in self._vrm_dev_keys)
                    and (self._dev_id not in self._vrm_dev_handles)
                )
                return False
            # If discovered but not connected
            elif self._vrm_dev_refs[self._dev_id] == 0:
                assert(
                    (self._dev_id not in self._vrm_dev_keys)
                    and (self._dev_id not in self._vrm_dev_handles)
                )
                return False
            # If discovered and connected
            elif self._vrm_dev_refs[self._dev_id] > 0:
                assert(
                    (self._dev_id in self._vrm_dev_keys)
                    and (self._dev_id in self._vrm_dev_handles)
                )
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

    @staticmethod
    def _get_vrm_dev_key_list():
        """
        Uses the vrmusbcam2 API to get the vrm device key list.

        Make sure to free the device keys using `vrm.VRmUsbCamFreeDeviceKey`
        immediately after usage!

        Returns
        -------
        dev_keys : `list(vrm.VRmDeviceKey)`
            List of vrm device keys.
        """
        # Update available devices
        vrm.VRmUsbCamUpdateDeviceKeyList()
        dev_count = vrm.VRmDWORD()
        vrm.VRmUsbCamGetDeviceKeyListSize(ct.byref(dev_count))
        dev_count = int(dev_count.value)
        dev_keys = [vrm.POINTER(vrm.VRmDeviceKey)() for _ in range(dev_count)]
        # Get device keys and extract device IDs
        [
            vrm.VRmUsbCamGetDeviceKeyListEntry(i, ct.byref(dev_key))
            for i, dev_key in enumerate(dev_keys)
        ]
        return dev_keys

    @classmethod
    def discover(cls):
        dev_keys = cls._get_vrm_dev_key_list()
        dev_ids = [dev_key.contents.m_serial for dev_key in dev_keys]
        for dev_key in dev_keys:
            vrm.VRmUsbCamFreeDeviceKey(dev_key)
        del dev_keys
        # Check for devices which have become unavailable
        for id in list(cls._vrm_dev_refs):
            if id not in dev_ids:
                del cls._vrm_dev_refs[id]
                if id in cls._vrm_dev_keys:
                    cls.LOGGER.critical(
                        "device lost connection ({:s})".format(id)
                    )
                    del cls._vrm_dev_keys[id]
                    del cls._vrm_dev_handles[id]
                    # TODO: notify affected devices
        # Check for devices which have been added
        for id in dev_ids:
            if id not in cls._vrm_dev_refs:
                cls._vrm_dev_refs[id] = 0
        return dev_ids


###############################################################################
# Device
###############################################################################


class VRmagicVRmCX(Camera):

    def __init__(self):
        super().__init__()
        self.properties.set_properties(**self._get_default_properties_dict(
            "device_name"
        ))

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    def setup(self):
        if not isinstance(self.interface, ItfVRmagic):
            self.interface = ItfVRmagic()
        self.interface.setup()

    def shutdown(self):
        if self.is_connected():
            self.close()
        self.interface.shutdown()

    def is_set_up(self):
        return self.interface.is_set_up()

    def connect(self):
        if not self.is_set_up():
            raise RuntimeError("device not set up")
        self.interface.connect(self.identifier)
        self.interface.register(self.identifier, self)
        self.read_device_name()
        self.read_acquisition_frames()
        self.p.read_all()

    def close(self):
        if self.is_connected():
            self.interface.deregister(id=self.identifier)
            self.interface.close()

    def is_connected(self):
        return self.interface.is_connected()

    # ++++++++++++++++++++++++++++++++++++++++
    # Camera methods
    # ++++++++++++++++++++++++++++++++++++++++

    def _start_acquisition(self):
        vrm.VRmUsbCamStart(self.vrm_dev_handle)

    def _end_acquisition(self):
        vrm.VRmUsbCamStop(self.vrm_dev_handle)

    def next(self):
        vrm_image = vrm.POINTER(vrm.VRmImage)()
        vrm_frames_dropped = vrm.VRmBOOL()
        vrm.VRmUsbCamLockNextImage(
            self.vrm_dev_handle,
            ct.byref(vrm_image), ct.byref(vrm_frames_dropped)
        )
        height = vrm_image.contents.m_image_format.m_height
        pitch = vrm_image.contents.m_pitch
        np_image = (
            np.array(vrm_image.contents.mp_buffer[0:height*pitch])
            .reshape(height, pitch)
        )
        vrm.VRmUsbCamUnlockNextImage(self.vrm_dev_handle, ct.byref(vrm_image))
        # Save to buffer
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put(np_image)
        return np_image

    # ++++++++++++++++++++++++++++++++++++++++
    # Helper methods
    # ++++++++++++++++++++++++++++++++++++++++

    @property
    def vrm_dev_handle(self):
        return self.interface._vrm_dev_handles[self.identifier]

    @property
    def vrm_dev_key_contents(self):
        return self.interface._vrm_dev_keys[self.identifier].contents

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_pixel_hrzt_count(self):
        MAP = {
            "VRmC-12/BW": 754,
            "VRmC-9/BW": 1288,
            "VRmC-9+/BW": 1288,
        }
        value = MAP[self.p.device_name]
        self.p.pixel_hrzt_count = value
        return value

    def write_pixel_hrzt_count(self, value):
        if value != self.p.pixel_hrzt_count:
            self.LOGGER.warning("cannot write pixel_hrzt_count")

    def read_pixel_hrzt_size(self):
        MAP = {
            "VRmC-12/BW": 6.0e-6,
            "VRmC-9/BW": 5.2e-6,
            "VRmC-9+/BW": 5.2e-6,
        }
        value = MAP[self.p.device_name]
        self.p.pixel_hrzt_size = value
        return value

    def write_pixel_hrzt_size(self, value):
        if value != self.p.pixel_hrzt_size:
            self.LOGGER.warning("cannot write pixel_hrzt_size")

    def read_pixel_hrzt_offset(self):
        value = 0
        self.p.pixel_hrzt_offset = value
        return value

    def write_pixel_hrzt_offset(self, value):
        if value != self.p.pixel_hrzt_offset:
            self.LOGGER.warning("cannot write pixel_hrzt_offset")

    def read_pixel_vert_count(self):
        MAP = {
            "VRmC-12/BW": 482,
            "VRmC-9/BW": 1032,
            "VRmC-9+/BW": 1032,
        }
        value = MAP[self.p.device_name]
        self.p.pixel_vert_count = value
        return value

    def write_pixel_vert_count(self, value):
        if value != self.p.pixel_vert_count:
            self.LOGGER.warning("cannot write pixel_hrzt_count")

    def read_pixel_vert_size(self):
        MAP = {
            "VRmC-12/BW": 6.0e-6,
            "VRmC-9/BW": 5.2e-6,
            "VRmC-9+/BW": 5.2e-6,
        }
        value = MAP[self.p.device_name]
        self.p.pixel_vert_size = value
        return value

    def write_pixel_vert_size(self, value):
        if value != self.p.pixel_vert_size:
            self.LOGGER.warning("cannot write pixel_vert_size")

    def read_pixel_vert_offset(self):
        value = 0
        self.p.pixel_vert_offset = value
        return value

    def write_pixel_vert_offset(self, value):
        if value != self.p.pixel_vert_offset:
            self.LOGGER.warning("cannot write pixel_vert_offset")

    def read_format_color(self):
        value = FORMAT_COLOR.GS
        self.format_color = value
        return value

    def write_format_color(self, value):
        if value != self.p.format_color:
            self.LOGGER.warning("cannot write format_color")

    def read_channel_bitdepth(self):
        value = 8
        self.channel_bitdepth = value
        return value

    def write_channel_bitdepth(self, value):
        if value != self.p.channel_bitdepth:
            self.LOGGER.warning("cannot write channel_bitdepth")

    def read_exposure_mode(self):
        ct_val = ct.c_bool()
        vrm.VRmUsbCamGetPropertyValueB(
            self.vrm_dev_handle,
            vrm.VRM_PROPID_CAM_AUTO_EXPOSURE_B,
            ct.byref(ct_val)
        )
        MAP = {True: EXPOSURE_MODE.CONTINUOS, False: EXPOSURE_MODE.MANUAL}
        value = MAP[ct_val.value]
        self.p.exposure_mode = value
        return value

    def write_exposure_mode(self, value):
        MAP = {EXPOSURE_MODE.MANUAL: False, EXPOSURE_MODE.CONTINUOS: True}
        ct_val = ct.c_bool(MAP[value])
        vrm.VRmUsbCamSetPropertyValueB(
            self.vrm_dev_handle,
            vrm.VRM_PROPID_CAM_AUTO_EXPOSURE_B,
            ct.byref(ct_val)
        )
        self.p.exposure_mode = value

    def read_exposure_time(self):
        ct_val = ct.c_float()
        vrm.VRmUsbCamGetPropertyValueF(
            self.vrm_dev_handle,
            vrm.VRM_PROPID_CAM_EXPOSURE_TIME_F,
            ct.byref(ct_val)
        )
        value = ct_val.value / 1e3
        self.p.exposure_time = value
        return value

    def write_exposure_time(self, value):
        ct_val = ct.c_float(value * 1e3)
        vrm.VRmUsbCamSetPropertyValueF(
            self.vrm_dev_handle,
            vrm.VRM_PROPID_CAM_EXPOSURE_TIME_F,
            ct.byref(ct_val)
        )
        self.p.exposure_time = value

    def read_acquisition_frames(self):
        if self.p.acquisition_frames is None:
            self.p.acquisition_frames = 0
        return self.p.acquisition_frames

    def write_acquisition_frames(self, value):
        self.p.acquisition_frames = value

    def read_device_name(self):
        name = self.vrm_dev_key_contents.mp_product_str.data.decode("utf-8")
        self.p.device_name = name
        return name

    def write_device_name(self, value):
        if value != self.p.device_name:
            self.LOGGER.warning("cannot write device_name")
