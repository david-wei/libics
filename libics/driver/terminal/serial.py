import glob
import serial
import sys

from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal


###############################################################################


class BYTESIZE:

    FIVEBITS = serial.FIVEBITS
    SIXBITS = serial.SIXBITS
    SEVENBITS = serial.SEVENBITS
    EIGHTBITS = serial.EIGHTBITS


class PARITY:

    NONE = serial.PARITY_NONE
    EVEN = serial.PARITY_EVEN
    ODD = serial.PARITY_ODD
    MARK = serial.PARITY_MARK
    SPACE = serial.PARITY_SPACE


class STOPBITS:

    ONE = serial.STOPBITS_ONE
    ONE_POINT_FIVE = serial.STOPBITS_ONE_POINT_FIVE
    TWP = serial.STOPBITS_TWO


###############################################################################


class ItfSerial(ItfTerminal):

    """
    Parameters
    ----------
    baudrate : `int`
        Baud rate in bits per second.
    bytesize : `BYTESIZE`
        Number of data bits.
    parity : `PARITY`
        Parity checking.
    stopbits : `STOPBITS`
        Number of stop bits.
    """

    BAUDRATE = 115200
    BYTESIZE = BYTESIZE.EIGHTBITS
    PARITY = PARITY.NONE
    STOPBITS = STOPBITS.ONE

    def __init__(self):
        super().__init__()
        self._serial = None
        self.baudrate = self.BAUDRATE
        self.bytesize = self.BYTESIZE
        self.parity = self.PARITY
        self.stopbits = self.STOPBITS

    def setup(self):
        self._serial = serial.Serial(
            port=self.address,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=self.recv_timeout,
            write_timeout=self.send_timeout
        )

    def shutdown(self):
        self.close()
        self._serial = None
        for dev in self.devices():
            self.deregister(dev)

    def is_set_up(self):
        return self._serial is not None

    def connect(self):
        if not self._serial.is_open:
            self._serial.open()

    def close(self):
        if self.is_connected():
            self._serial.close()

    def is_connected(self):
        return self._serial.is_open

    @classmethod
    def discover(cls):
        """
        Discovers serial addresses.

        Returns
        -------
        result : `list(str)`
            List of addresses, e.g. `["COM3"]`.

        Notes
        -----
        From:
        https://stackoverflow.com/questions/12090503/listing-available-com-ports-with-python
        """
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif (
            sys.platform.startswith('linux')
            or sys.platform.startswith('cygwin')
        ):
            # this excludes your current terminal "/dev/tty"
            ports = glob.glob('/dev/tty[A-Za-z]*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/tty.*')
        else:
            raise EnvironmentError('Unsupported platform')

        result = []
        for port in ports:
            try:
                s = serial.Serial(port)
                s.close()
                result.append(port)
            except (OSError, serial.SerialException):
                pass
        return result

    def status(self):
        status = STATUS()
        if self.is_set_up():
            if self.is_connected():
                status.set_state(STATUS.OK)
            else:
                status.set_state(STATUS.ERROR)
                status.set_err_type(STATUS.ERR_INSTANCE)
        else:
            status.set_state(STATUS.OK)
        return status

    # ++++++++++++++++++++++++++++++++++++++++

    def send(self, s_data):
        s_data = str(s_data) + self.send_termchar
        self.LOGGER.debug("SEND: {:s}".format(s_data))
        b_data = s_data.encode("ascii")
        self._serial.write(b_data)
        self._serial.flush()

    def recv(self):
        b_data = self._serial.read_until(
            self.recv_termchar.encode("ascii")
        )
        s_data = b_data.decode("ascii")
        self.LOGGER.debug("RECV: {:s}".format(s_data))
        s_data = self._trim(s_data)
        return s_data

    def recv_waiting(self):
        b_data = self._serial.read(size=self._serial.in_waiting)
        s_data = b_data.decode("ascii")
        self.LOGGER.debug("RECV: {:s}".format(s_data))
        s_data = self._trim(s_data)
        return s_data

    def flush_out(self):
        self._serial.reset_output_buffer()

    def flush_in(self):
        self._serial.reset_input_buffer()
