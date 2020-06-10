
class VRmagicItf():

    def __init__(self, cfg):
        super().__init__(cfg)
        self._dev_key = None
        self._dev_handle = None
        self._buffer_bool = vrm.VRmBOOL()
        self._buffer_double = vrm.c_double()
        self._buffer_int = ctypes.c_int()
        self._buffer_float = ctypes.c_float()
        self._buffer_e = vrm.VRmPropId()
        self._buffer_str = ctypes.c_char_p()
        self._buffer_frames_dropped = vrm.VRmBOOL()
        self._buffer_img_ready = vrm.VRmBOOL()
        self._buffer_img_data = vrm.POINTER(vrm.VRmImage)()

    def setup(self):
        # Initialize VRmagic API
        vrm.VRmUsbCamUpdateDeviceKeyList()
        dev_count = vrm.VRmDWORD()
        vrm.VRmUsbCamGetDeviceKeyListSize(ctypes.byref(dev_count))
        dev_count = int(dev_count.value)
        # Get VRmagic device key
        dev_keys = [vrm.POINTER(vrm.VRmDeviceKey)() for _ in range(dev_count)]
        [vrm.VRmUsbCamGetDeviceKeyListEntry(i, ctypes.byref(dev_key))
         for i, dev_key in enumerate(dev_keys)]
        if self.cfg.device is None:
            if dev_count == 0:
                raise NameError("No VRmagic camera found")
            elif dev_count > 1:
                print("Warning: multiple VRmagic cameras found")
            self._dev_key = dev_keys[0]
            for k in dev_keys[1:]:
                vrm.VRmUsbCamFreeDeviceKey(k)
        else:
            for k in dev_keys:
                if k.contents.m_serial == self.cfg.device:
                    self._dev_key = k
                else:
                    vrm.VRmUsbCamFreeDeviceKey(k)
        if self._dev_key is None:
            raise NameError("No suitable VRmagic camera found")

    def shutdown(self):
        vrm.VRmUsbCamFreeDeviceKey(self._dev_key)
        self._dev_key = None

    def connect(self):
        self._dev_handle = vrm.VRmUsbCamDevice()
        vrm.VRmUsbCamOpenDevice(self._dev_key, ctypes.byref(self._dev_handle))

    def close(self):
        vrm.VRmUsbCamCloseDevice(self._dev_handle)
        self._dev_handle = None



###############################################################################


class VRmagicVRmCX(CamDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._frame_buffer = None
        self._frame_buffer_lock = threading.Lock()
        self._callback = None
        self._thread_continuos_acquisition = None
        self._exposure_mode = drv.DRV_CAM.EXPOSURE_MODE.MANUAL

    def run(self, callback=None):
        self._callback = callback
        vrm.VRmUsbCamStart(self._interface._dev_handle)
        if self._exposure_mode == drv.DRV_CAM.EXPOSURE_MODE.CONTINUOS:
            self._thread_continuos_acquisition = util.thread.StoppableThread()
            self._thread_continuos_acquisition.run = self._acquisition_loop
            self._thread_continuos_acquisition.start()

    def stop(self):
        if self._thread_continuos_acquisition is not None:
            self._thread_continuos_acquisition.stop()
        self._thread_continuos_acquisition = None
        self._callback = None
        vrm.VRmUsbCamStop(self._interface._dev_handle)

    def grab(self):
        vrm.VRmUsbCamLockNextImage(
            self._interface._dev_handle,
            ct.byref(self._interface._buffer_img_data),
            ct.byref(self._interface._buffer_frames_dropped)
        )
        h = self._interface._buffer_img_data.contents.m_image_format.m_height
        p = self._interface._buffer_img_data.contents.m_pitch
        im = np.array(
            self._interface._buffer_img_data.contents.mp_buffer[0:h*p]
        ).reshape(h, p)
        vrm.VRmUsbCamUnlockNextImage(
            self._interface._dev_handle,
            ct.byref(self._interface._buffer_img_data)
        )
        return im

    def get(self):
        self._frame_buffer_lock.acquire()
        im = np.copy(self._frame_buffer)
        self._frame_buffer_lock.release()
        return im

    def _acquisition_loop(self):
        while not (
            self._thread_continuos_acquisition.stop_event
            .wait(timeout=self.cfg.exposure_time.val / 3)
        ):
            self._frame_buffer_lock.acquire()
            im = self.grab()
            self._frame_buffer = im
            if self._callback is not None:
                self._callback(np.copy(im))
            self._frame_buffer_lock.release()

    # ++++ Write/read methods +++++++++++

    def _read_dev_name(self):
        name = self._interface._dev_key.contents.mp_product_str.data
        return name.decode("utf-8")

    def _read_pixel_hrzt_count(self):
        name = self._read_dev_name()
        MAP = {
            "VRmC-12/BW": 754,
            "VRmC-9/BW": 1288,
            "VRmC-9+/BW": 1288,
        }
        return MAP[name]

    def _read_pixel_hrzt_size(self):
        name = self._read_dev_name()
        MAP = {
            "VRmC-12/BW": 6.0,
            "VRmC-9/BW": 5.2,
            "VRmC-9+/BW": 5.2,
        }
        return MAP[name]

    def _read_pixel_hrzt_offset(self):
        return 0

    def _read_pixel_vert_count(self):
        name = self._read_dev_name()
        MAP = {
            "VRmC-12/BW": 482,
            "VRmC-9/BW": 1032,
            "VRmC-9+/BW": 1032,
        }
        return MAP[name]

    def _read_pixel_vert_size(self):
        name = self._read_dev_name()
        MAP = {
            "VRmC-12/BW": 6.0,
            "VRmC-9/BW": 5.2,
            "VRmC-9+/BW": 5.2,
        }
        return MAP[name]

    def _read_pixel_vert_offset(self):
        return 0

    def _read_format_color(self):
        return drv.DRV_CAM.FORMAT_COLOR.GS

    def _read_channel_bitdepth(self):
        return 8

    def _write_exposure_mode(self, value):
        if value not in [
            drv.DRV_CAM.EXPOSURE_MODE.MANUAL,
            drv.DRV_CAM.EXPOSURE_MODE.CONTINUOS,
        ]:
            raise KeyError("invalid exposure mode")
        self._exposure_mode = value

    def _read_exposure_mode(self):
        return self._exposure_mode

    def _write_exposure_time(self, value):
        ct_val = ct.c_float(value * 1e3)
        vrm.VRmUsbCamSetPropertyValueF(
            self._interface._dev_handle,
            vrm.VRM_PROPID_CAM_EXPOSURE_TIME_F,
            ct.byref(ct_val)
        )

    def _read_exposure_time(self):
        ct_val = ct.c_float()
        vrm.VRmUsbCamGetPropertyValueF(
            self._interface._dev_handle,
            vrm.VRM_PROPID_CAM_EXPOSURE_TIME_F,
            ct.byref(ct_val)
        )
        return ct_val.value / 1e3
