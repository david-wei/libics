# System Imports
import select
import six
import socket
import time


###############################################################################


class ConnectionGpibEthernet(object):

    def __init__(self, ipv4, port):
        self.ipv4 = ipv4
        self.port = port
        self.socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_STREAM
        )

    def connect(self):
        self.socket.connect((self.ipv4, self.port))

    def close(self):
        self.socket.close()

    def send(self, data, termchar="\r\n"):
        data += termchar
        if six.PY3:
            data = bytes(data, "ascii")
        return self.socket.send(data)

    @staticmethod
    def _has_termchar(buffer, termchar="\n"):
        buffer = buffer[-len(termchar):]
        if six.PY3:
            buffer = buffer.decode("ascii")
        return buffer == termchar

    def receive(self,
                data_timeout=0.1, receiver_timeout=5.0,
                buffer_size=4096, termchar="\n"):
        self.socket.setblocking(0)
        data = []
        t0 = time.time()

        while True:
            ready = select.select([self.socket], [], [], data_timeout)
            if ready[0]:
                buffer = self.socket.recv(buffer_size)
                data.append(buffer)
            if ConnectionGpibEthernet._has_termchar(buffer, termchar=termchar):
                break
            if time.time() - t0 > receiver_timeout:
                break

        self.socket.set_blocking(1)
        if six.PY3:
            data = [item.decode("ascii") for item in data]
        data = "".join(data)
        return data[0:len(data) - len(termchar)]


class ConnectionGpibUsb(object):

    pass


###############################################################################


class GPIB(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.data = []
        self._buffer_size = 4096
        self._sender_termchar = "\r\n"
        self._data_timeout = 0.1
        self._receiver_timeout = 5.0
        self._receiver_termchar = "\n"

    # ++++ Configuration +++++++++++++++++++++

    def set_connection(self, connection=None):
        if connection is not None:
            self.connection = connection

    def set_sender(self, termchar=None):
        if termchar is not None:
            self._sender_termchar = termchar

    def set_receiver(self,
                     data_timeout=None, receiver_timeout=None,
                     termchar=None, buffer_size=None):
        if data_timeout is not None:
            self._data_timeout = data_timeout
        if receiver_timeout is not None:
            self._receiver_timeout = receiver_timeout
        if termchar is not None:
            self._receiver_termchar = termchar
        if buffer_size is not None:
            self._buffer_size = buffer_size

    # ++++ Connection ++++++++++++++++++++++++

    def connect(self):
        return self.connection.connect()

    def close(self):
        return self.connection.close()

    # ++++ Communication +++++++++++++++++++++

    def send(self, data):
        return self.connection.send(data, termchar=self._sender_termchar)

    def receive(self):
        self.data = self.connection.receive(
            data_timeout=self._data_timeout,
            receiver_timeout=self._receiver_timeout,
            buffer_size=self._buffer_size, termchar=self._receiver_termchar
        )
        return self.data
