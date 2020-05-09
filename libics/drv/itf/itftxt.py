# System Imports
import abc
import select
import serial
import socket
import time
import usb.core
import usb.util

# Package Imports
from libics.drv.itf import itf


###############################################################################


def get_txt_itf(cfg):
    if cfg.interface == itf.ITF_TXT.SERIAL:
        return TxtSerialItf(cfg)
    elif cfg.interface == itf.ITF_TXT.ETHERNET:
        return get_txt_ethernet_itf(cfg)
    elif cfg.interface == itf.ITF_TXT.USB:
        return TxtUsbItf(cfg)


class TxtItfBase(abc.ABC):

    def __init__(self, cfg=None):
        self.cfg = cfg

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    @abc.abstractmethod
    def connect(self):
        return self.connection.connect()

    @abc.abstractmethod
    def close(self):
        return self.connection.close()

    @abc.abstractmethod
    def send(self, s_data):
        return self.connection.send()

    @abc.abstractmethod
    def recv(self):
        return self.connection.recv()

    def query(self, s_data):
        self.send(s_data)
        return self.recv()


###############################################################################


class TxtSerialItf(TxtItfBase):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self._serial = None

    def setup(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg
        _bytesize = {
            5: serial.FIVEBITS,
            6: serial.SIXBITS,
            7: serial.SEVENBITS,
            8: serial.EIGHTBITS
        }
        _parity = {
            "none": serial.PARITY_NONE,
            "even": serial.PARITY_EVEN,
            "odd": serial.PARITY_ODD,
            "mark": serial.PARITY_MARK,
            "space": serial.PARITY_SPACE
        }
        _stopbits = {
            1: serial.STOPBITS_ONE,
            1.5: serial.STOPBITS_ONE_POINT_FIVE,
            2: serial.STOPBITS_TWO
        }
        self._serial = serial.Serial(
            port=self.cfg.address,
            baudrate=self.cfg.baudrate,
            bytesize=_bytesize[self.cfg.bytesize],
            parity=_parity[self.cfg.parity],
            stopbits=_stopbits[self.cfg.stopbits],
            timeout=self.cfg.recv_timeout,
            write_timeout=self.cfg.send_timeout
        )

    def shutdown(self):
        self.close()

    def connect(self):
        if not self._serial.is_open:
            self._serial.open()
        return self._serial.is_open

    def close(self):
        if self._serial.is_open:
            self._serial.close()
        return not self._serial.is_open

    def send(self, s_data):
        s_data = str(s_data) + self.cfg.send_termchar
        b_data = s_data.encode("ascii")
        self._serial.write(b_data)
        self._serial.flush()

    def recv(self):
        b_data = self._serial.readline()
        s_data = b_data.decode("ascii").strip(self.cfg.recv_termchar)
        return s_data


def get_txt_ethernet_itf(cfg):
    if cfg.txt_ethernet_type == itf.TXT_ETHERNET_TYPE.GENERIC:
        return TxtEthernetItf(cfg)
    elif cfg.txt_ethernet_type == itf.TXT_ETHERNET_TYPE.GPIB:
        if cfg.ctrl_model == itf.TXT_ETHERNET_GPIB.MODEL.GENERIC:
            return TxtEthernetItf(cfg)
        elif (cfg.ctrl_model ==
              itf.TXT_ETHERNET_GPIB.MODEL.PROLOGIX_GPIB_ETHERNET):
            return PrologixGpibEthernetItf(cfg)


class TxtEthernetItf(TxtItfBase):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self._socket = None

    def setup(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg
        self._socket = socket.socket(
            family=socket.AF_INET,
            type=socket.SOCK_STREAM
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
        self._socket.shutdown(socket.SHUT_RDWR)

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


class PrologixGpibEthernetItf(TxtEthernetItf):

    def __init__(self, cfg):
        super().__init__(cfg)

    def connect(self):
        super().connect()
        # Disable in-device cfg saving
        self.send("++savecfg 0")
        # Set GPIB mode
        GPIB_MODE = {
            itf.TXT_ETHERNET_GPIB.MODE.CONTROLLER: 1,
            itf.TXT_ETHERNET_GPIB.MODE.DEVICE: 0
        }
        self.send("++mode {:d}".format(GPIB_MODE[self.cfg.gpib_mode]))
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
        if self.cfg.gpib_mode == itf.TXT_ETHERNET_GPIB.MODE.CONTROLLER:
            self.send("++read")
        return super().recv()


class TxtUsbItf(TxtItfBase):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self._usb_dev = None
        self._usb_itf = None
        self._usb_ep_in = None
        self._usb_ep_out = None

    def setup(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg

    def shutdown(self):
        return True

    def connect(self):
        self._usb_dev = usb.core.find(
            idVendor=self.cfg.usb_vendor,
            idProduct=self.cfg.usb_product
        )
        self._usb_dev.set_configuration()
        self._usb_itf = self._usb_dev.get_active_configuration()[(0, 0)]
        self._usb_ep_in = usb.util.find_descriptor(
            self._usb_itf, custom_match=lambda e: (
                usb.util.endpoint_direction(e.bEndpointAddress)
                == usb.util.ENDPOINT_IN
            )
        )
        self._usb_ep_out = usb.util.find_descriptor(
            self._usb_itf, custom_match=lambda e: (
                usb.util.endpoint_direction(e.bEndpointAddress)
                == usb.util.ENDPOINT_OUT
            )
        )

    def close(self):
        usb.core.util.dispose_resources(self._usb_dev)
        self._usb_dev = None
        self._usb_itf = None
        self._usb_ep_in = None
        self._usb_ep_out = None

    def send(self, s_data):
        s_data = str(s_data) + self.cfg.send_termchar
        self._usb_ep_out.write(s_data, timeout=int(1000*self.cfg.send_timeout))

    def recv(self):
        ar_data = self._usb_ep_in.read(
            self.cfg.buffer_size, timeout=int(1000*self.cfg.recv_timeout)
        )
        s_data = "".join(chr(c) for c in ar_data)
        return s_data
