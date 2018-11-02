# System Imports
import abc
import select
import serial
import socket
import time

# Package Imports
from libics.drv.itf import itf


###############################################################################


def get_txt_itf(cfg):
    if cfg.interface == itf.ITF_TXT.SERIAL:
        return TxtSerialItf(cfg)
    elif cfg.interface == itf.ITF_TXT.ETHERNET:
        return TxtEthernetItf(cfg)


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
        self._socket.shutdown()
        return True

    def connect(self):
        try:
            self._socket.connect((self.cfg.address, self.cfg.port))
            return True
        except(socket.timeout, InterruptedError):
            return False

    def close(self):
        self._socket.close()
        return True

    def send(self, s_data):
        s_data = str(s_data) + self.cfg.send_termchar
        b_data = s_data.encode("ascii")
        self._socket.send(b_data)

    def recv(self):
        l_data = []
        s_data = ""
        t0 = time.time()
        len_recv_termchar = len(self.cfg.recv_termchar)
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
                if time.time() - t0 > self.cfg.recv_timeout:
                    break
        return s_data
