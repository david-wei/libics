import select
import socket
import time

from libics.env import logging
from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal


###############################################################################


class FAMILY:

    AF_INET = socket.AF_INET


class TYPE:

    SOCK_STREAM = socket.SOCK_STREAM


class SHUTDOWN:

    SHUT_RDWR = socket.SHUT_RDWR


###############################################################################


class ItfEthernet(ItfTerminal):

    """
    Parameters
    ----------
    port : `int`
        Ethernet port.
    blocking : `bool`
        Flag whether communication is blocking.
    """

    PORT = None
    BLOCKING = True

    LOGGER = logging.get_logger("libics.driver.terminal.ethernet.ItfEthernet")

    def __init__(self):
        super().__init__()
        self._socket = None
        self._is_connected = False
        self.port = self.PORT
        self.blocking = self.BLOCKING

    def setup(self):
        self._socket = socket.socket(
            family=FAMILY.AF_INET, type=TYPE.SOCK_STREAM
        )
        self._socket.settimeout(self.send_timeout)
        self._socket.setblocking(self.blocking)

    def shutdown(self):
        self._socket.close()
        self._socket = None
        for dev in self.devices():
            self.deregister(dev)

    def is_set_up(self):
        return self._socket is not None

    def connect(self):
        self._socket.connect((self.address, self.port))
        self._is_connected = True
        self.flush_in()

    def close(self):
        self._socket.shutdown(SHUTDOWN.SHUT_RDWR)
        self._is_connected = False

    def is_connected(self):
        return self._is_connected

    def discover(self):
        self.LOGGER.warning(
            "ethernet interface base class cannot discover devices"
        )
        return []

    def status(self):
        status = STATUS()
        if self.is_connected():
            try:
                self._socket.setblocking(0)
                self._socket.recv(self.buffer_size)
                self._socket.setblocking(self.blocking)
                status.set_state(STATUS.OK)
            except socket.timeout:
                status.set_state(STATUS.ERROR)
                status.set_err_type(STATUS.ERR_CONNECTION)
        else:
            status.set_state(STATUS.OK)
        return status

    # ++++++++++++++++++++++++++++++++++++++++

    def send(self, s_data):
        s_data = str(s_data) + self.send_termchar
        self.LOGGER.debug("SEND: {:s}".format(s_data))
        b_data = s_data.encode("ascii")
        self._socket.send(b_data)

    def recv(self):
        l_data = []
        s_data = ""
        t0 = time.time()
        len_recv_termchar = len(self.recv_termchar)
        self._socket.setblocking(0)
        while True:
            ready = select.select(
                [self._socket], [], [],
                self.send_timeout
            )
            if ready[0]:
                try:
                    b_buffer = self._socket.recv(self.buffer_size)
                except socket.timeout:
                    break
                s_buffer = b_buffer.decode("ascii")
                l_data.append(s_buffer)
                if s_buffer[-len_recv_termchar:] == self.recv_termchar:
                    l_data[-1] = l_data[-1][:-len_recv_termchar]
                    s_data = "".join(l_data)
                    break
            dt = time.time() - t0
            if dt > 10 * self.recv_timeout:
                break
        self._socket.setblocking(self.blocking)
        s_data = self._trim(s_data)
        self.LOGGER.debug("RECV: {:s}".format(s_data))
        return s_data

    def flush_out(self):
        # empty by default
        pass

    def flush_in(self):
        self._socket.setblocking(0)
        t0 = time.time()
        while True:
            ready = select.select(
                [self._socket], [], [],
                self.send_timeout
            )
            if ready[0]:
                self._socket.recv(self.buffer_size)
            else:
                break
            dt = time.time() - t0
            if dt > 10 * self.recv_timeout:
                break
