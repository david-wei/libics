
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

