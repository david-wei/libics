import usb.core
import usb.util

from libics.core.env import logging
from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal


###############################################################################


USBError = usb.core.USBError


class ENDPOINT:

    IN = usb.util.ENDPOINT_IN
    OUT = usb.util.ENDPOINT_OUT


###############################################################################


class ItfUsb(ItfTerminal):

    """
    Parameters
    ----------
    id_vendor : str
        USB vendor ID.
    id_product : str
        USB product ID.
    """

    ID_VENDOR = 0x0000
    ID_PRODUCT = 0x0000

    LOGGER = logging.get_logger("libics.driver.terminal.ethernet.ItfUsb")

    def __init__(self):
        super().__init__()
        self._dev = None
        self._itf = None
        self._ep_in = None
        self._ep_out = None
        # +++++++++++++++++++++++++
        self.id_vendor = self.ID_VENDOR
        self.id_product = self.ID_PRODUCT

    def setup(self):
        self._dev = usb.core.find(
            idVendor=self.id_vendor,
            idProduct=self.id_product
        )
        if self._dev is None:
            raise USBError("USB device not found.")

    def shutdown(self):
        usb.core.util.dispose_resources(self._dev)
        self._dev = None

    def is_setup(self):
        return self._dev is not None

    def connect(self):
        self._dev.reset()
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
        self._itf = None
        self._ep_in = None
        self._ep_out = None

    def is_connected(self):
        return self._itf is not None

    def discover(self):
        self.LOGGER.warning(
            "USB terminal interface base class cannot discover devices"
        )
        return []

    def status(self):
        # TODO: add proper status diagnosis
        status = STATUS()
        return status

    # ++++++++++++++++++++++++++++++++++++++++

    def send(self, s_data):
        s_data = str(s_data) + self.send_termchar
        self.LOGGER.debug("SEND: {:s}".format(s_data))
        self._ep_out.write(s_data, timeout=int(1000*self.send_timeout))

    def recv(self):
        ar_data = self._ep_in.read(
            self.buffer_size, timeout=int(1000*self.recv_timeout)
        )
        s_data = "".join(chr(c) for c in ar_data)
        self.LOGGER.debug("RECV: {:s}".format(s_data))
        return s_data

    def flush_out(self):
        # empty by default
        pass

    def flush_in(self):
        # empty by default
        pass
