import select
import socket
import time

from libics.driver.interface import ItfTerminal







class TXT_ETHERNET_TYPE:

    GENERIC = 0
    GPIB = 1


@InheritMap(map_key=("libics", "TxtEthernetCfg"))
class TxtEthernetCfg(TxtCfgBase):

    """
    ProtocolCfgBase -> TxtCfgBase -> TxtEthernetCfg.

    Parameters
    ----------
    txt_ethernet_type : TXT_ETHERNET_TYPE
        Type of text-based Ethernet interface.
    port : int
        Ethernet port number.
    blocking : bool
        Whether to block Ethernet socket.
    """

    def __init__(self, txt_ethernet_type=TXT_ETHERNET_TYPE.GENERIC,
                 port=None, blocking=False,
                 cls_name="TxtEthernetCfg", ll_obj=None, **kwargs):
        if "interface" not in kwargs.keys():
            kwargs["interface"] = ITF_TXT.ETHERNET
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.txt_ethernet_type = txt_ethernet_type
        self.port = port
        self.blocking = blocking

    def get_hl_cfg(self):
        if self.txt_ethernet_type == TXT_ETHERNET_TYPE.GPIB:
            return TxtEthernetGpibCfg(ll_obj=self, **self._kwargs)
        else:
            return self











class TXT_ETHERNET_GPIB:

    class MODE:

        CONTROLLER = 0
        DEVICE = 1

    class MODEL:

        GENERIC = 0
        PROLOGIX_GPIB_ETHERNET = 11


@InheritMap(map_key=("libics", "TxtEthernetGpibCfg"))
class TxtEthernetGpibCfg(TxtEthernetCfg):

    """
    ProtocolCfgBase -> TxtCfgBase -> TxtEthernetCfg -> TxtEthernetGpibCfg.

    Parameters
    ----------
    gpib_mode : TXT_ETHERNET_GPIB.MODE
        Whether device is in controller or device mode.
    gpib_address : int
        GPIB address.
    ctrl_model : TXT_ETHERNET_GPIB.MODEL
        Controller device model providing the interface.
    """

    def __init__(
        self,
        gpib_mode=TXT_ETHERNET_GPIB.MODE, gpib_address=1,
        ctrl_model=TXT_ETHERNET_GPIB.MODEL.GENERIC,
        cls_name="TxtEthernetGpibCfg", ll_obj=None, **kwargs
    ):
        if "txt_ethernet_type" not in kwargs.keys():
            kwargs["txt_ethernet_type"] = TXT_ETHERNET_TYPE.GPIB
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.gpib_mode = gpib_mode
        self.gpib_address = gpib_address
        self.ctrl_model = ctrl_model

    def get_hl_cfg(self):
        return self





###############################################################################


class FAMILY:

    AF_INET = socket.AF_INET


class TYPE:

    SOCK_STREAM = socket.SOCK_STREAM


class SHUTDOWN:

    SHUT_RDWR = socket.SHUT_RDWR


###############################################################################


class ItfEthernet(ItfTerminal):

    def __init__(self):
        super().__init__()
        self._socket = None

    def setup(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg
        self._socket = socket.socket(
            family=FAMILY.AF_INET, type=TYPE.SOCK_STREAM
        )
        self._socket.settimeout(self.cfg.send_timeout)
        self._socket.setblocking(self.cfg.blocking)

    def shutdown(self):
        self._socket.close()
        return True

    def connect(self):
        self._socket.connect((self.cfg.address, self.cfg.port))
        self.empty_buffer()

    def close(self):
        self._socket.shutdown(SHUTDOWN.SHUT_RDWR)

    def send(self, s_data):
        s_data = str(s_data) + self.cfg.send_termchar
        b_data = s_data.encode("ascii")
        self._socket.send(b_data)

    def recv(self):
        l_data = []
        s_data = ""
        t0 = time.time()
        len_recv_termchar = len(self.cfg.recv_termchar)
        self._socket.setblocking(0)
        while True:
            ready = select.select(
                [self._socket], [], [],
                self.cfg.send_timeout
            )
            if ready[0]:
                b_buffer = self._socket.recv(self.cfg.buffer_size)
                s_buffer = b_buffer.decode("ascii")
                l_data.append(s_buffer)
                if s_buffer[-len_recv_termchar:] == self.cfg.recv_termchar:
                    l_data[-1] = l_data[-1][:-len_recv_termchar]
                    s_data = "".join(l_data)
                    break
            dt = time.time() - t0
            if dt > 10 * self.cfg.recv_timeout:
                break
        self._socket.setblocking(self.cfg.blocking)
        return s_data

    def empty_buffer(self):
        self._socket.setblocking(0)
        t0 = time.time()
        while True:
            ready = select.select(
                [self._socket], [], [],
                self.cfg.send_timeout
            )
            if ready[0]:
                self._socket.recv(self.cfg.buffer_size)
            else:
                break
            dt = time.time() - t0
            if dt > 10 * self.cfg.recv_timeout:
                break


class ItfEthernetPrologixGpib(ItfEthernet):

    class GPIB_MODE:

        CONTROLLER = "CONTROLLER"
        DEVICE = "DEVICE"

    def __init__(self, cfg):
        super().__init__(cfg)

    def connect(self):
        super().connect()
        # Disable in-device cfg saving
        self.send("++savecfg 0")
        # Set GPIB mode
        _MODE = {self.GPIB_MODE.CONTROLLER: 1, self.GPIB_MODE.DEVICE: 0}
        self.send("++mode {:d}".format(_MODE[self.cfg.gpib_mode]))
        # Enable EOI assertion
        self.send("++eoi 1")
        # Disable automatic read after write mode
        self.send("++auto 0")
        # Disable auto-appending termchars
        self.send("++eos 3")
        # Set GPIB address
        self.send("++addr {:d}".format(self.cfg.gpib_address))
        # Set read timeout
        self.send("++read_tmo_ms {:d}"
                  .format(round(1000 * self.cfg.recv_timeout)))
        time.sleep(2)

    def close(self):
        # Set local
        self.send("++loc")
        super().close()

    def recv(self):
        if self.cfg.gpib_mode == self.GPIB_MODE.CONTROLLER:
            self.send("++read")
        return super().recv()
