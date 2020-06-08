import abc

from libics.driver.interface import ItfBase


###############################################################################


class ItfTerminal(ItfBase):

    """
    Parameters
    ----------
    address : `str`
        Address of interface, e.g. IP.
    buffer_size : `int`
        Number of buffer bytes.
    send_timeout : `float`
        Send: timeout in seconds.
    send_termchar : `str`
        Send: command termination characters.
    recv_timeout : `float`
        Receive: timeout in seconds.
    recv_termchar : `str`
        Receive: command termination characters.
    """

    ADDRESS = ""
    BUFFER_SIZE = 1024
    SEND_TIMEOUT = 1.0
    SEND_TERMCHAR = "\r\n"
    RECV_TIMEOUT = 1.0
    RECV_TERMCHAR = "\r\n"

    def __init__(self):
        super().__init__()
        self.address = self.ADDRESS
        self.buffer_size = self.BUFFER_SIZE
        self.send_timeout = self.SEND_TIMEOUT
        self.send_termchar = self.SEND_TERMCHAR
        self.recv_timeout = self.RECV_TIMEOUT
        self.recv_termchar = self.RECV_TERMCHAR

    @abc.abstractmethod
    def send(self, msg):
        """
        Sends a message.

        Parameters
        ----------
        msg : `str`
            Message string.
        """

    @abc.abstractmethod
    def recv(self):
        """
        Receives a message.

        Returns
        -------
        msg : `str`
            Message string.
        """

    def query(self, msg):
        """
        Sends and receives a message.

        Parameters
        ----------
        msg : `str`
            Message string to be sent.

        Returns
        -------
        msg : `str`
            Message string received.
        """
        self.send(msg)
        return self.recv()

    def validate(self, msg, val):
        """
        Sends a message and checks for validation response.

        Parameters
        ----------
        msg : `str`
            Message string to be sent.
        val : `str`
            Validation string with which the response is compared.

        Returns
        -------
        val : `bool`
            Flag whether command is validated.
        """
        self.send(msg)
        return self.recv() == val

    def _trim(self, msg):
        """
        Pre-processes a received message.

        Parameters
        ----------
        msg : `str`
            Raw message string.

        Returns
        -------
        msg : `str`
            Trimmed message string.
        """
        return msg.strip(self.recv_termchar)

    @abc.abstractmethod
    def flush_out(self):
        """
        Flushes the output buffer.
        """

    @abc.abstractmethod
    def flush_in(self):
        """
        Flushes the input buffer.
        """


###############################################################################


from . import ethernet
from . import gpib
from . import serial
from . import usb
