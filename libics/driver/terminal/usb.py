import usb.core
import usb.util

from libics.driver.interface import ItfTerminal





@InheritMap(map_key=("libics", "TxtUsbCfg"))
class TxtUsbCfg(TxtCfgBase):

    """
    ProtocolCfgBase -> TxtCfgBase -> TxtUsbCfg.

    Parameters
    ----------
    usb_vendor : str
        USB vendor ID.
    usb_product : str
        USB product ID.
    """

    def __init__(self, usb_vendor=0x104d, usb_product=0x4000,
                 cls_name="TxtUsbCfg", ll_obj=None, **kwargs):
        if "interface" not in kwargs.keys():
            kwargs["interface"] = ITF_TXT.USB
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.usb_vendor = usb_vendor
        self.usb_product = usb_product

    def get_hl_cfg(self):
        return self









###############################################################################


USBError = usb.core.USBError


class ENDPOINT:

    IN = usb.util.ENDPOINT_IN
    OUT = usb.util.ENDPOINT_OUT


###############################################################################


class ItfUsb(ItfTerminal):

    def __init__(self):
        super().__init__()
        self._dev = None
        self._itf = None
        self._ep_in = None
        self._ep_out = None
        # +++++++++++++++++++++++++
        self.id_vendor = None
        self.id_product = None

    def setup(self):
        pass

    def shutdown(self):
        pass

    def connect(self):
        self._dev = usb.core.find(
            idVendor=self.id_vendor,
            idProduct=self.id_product
        )
        if self._dev is None:
            raise USBError("USB device not found.")
        self._dev.set_configuration()
        self._itf = self._usb_dev.get_active_configuration()[(0, 0)]
        self._ep_in = usb.util.find_descriptor(
            self._itf, custom_match=lambda e: (
                usb.util.endpoint_direction(e.bEndpointAddress)
                == ENDPOINT.IN
            )
        )
        self._ep_out = usb.util.find_descriptor(
            self._itf, custom_match=lambda e: (
                usb.util.endpoint_direction(e.bEndpointAddress)
                == ENDPOINT.OUT
            )
        )

    def close(self):
        usb.core.util.dispose_resources(self._dev)
        self._dev = None
        self._itf = None
        self._ep_in = None
        self._ep_out = None

    def send(self, s_data):
        s_data = str(s_data) + self.cfg.send_termchar
        self._ep_out.write(s_data, timeout=int(1000*self.send_timeout))

    def recv(self):
        ar_data = self._ep_in.read(
            self.buffer_size, timeout=int(1000*self.recv_timeout)
        )
        s_data = "".join(chr(c) for c in ar_data)
        return s_data
