import serial

from libics.driver.interface import STATUS
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

    def is_setup(self):
        return self._serial is not None

    def connect(self):
        if not self._serial.is_open:
            self._serial.open()

    def close(self):
        if self.is_connected():
            self._serial.close()

    def is_connected(self):
        return self._serial.is_open

    def discover(self):
        self.LOGGER.warning(
            "serial interface base class cannot discover devices"
        )
        return []

    def status(self):
        status = {STATUS.MSG: ""}
        if self.is_setup():
            if self.is_connected():
                status[STATUS.OK] = ""
            else:
                status[STATUS.ERROR] = ""
                status[STATUS.ERR_CONNECTION] = ""
                status[STATUS.ERR_INSTANCE] = ""
        else:
            status[STATUS.OK] = ""
        return status

    # ++++++++++++++++++++++++++++++++++++++++

    def send(self, s_data):
        s_data = str(s_data) + self.cfg.send_termchar
        self.LOGGER.debug("SEND: {:s}".format(s_data))
        b_data = s_data.encode("ascii")
        self._serial.write(b_data)
        self._serial.flush()

    def recv(self):
        b_data = self._serial.readline()
        s_data = b_data.decode("ascii")
        self.LOGGER.debug("RECV: {:s}".format(s_data))
        s_data = self._trim(s_data)
        return s_data

    def flush_out(self):
        self._serial.reset_output_buffer()

    def flush_in(self):
        self._serial.reset_input_buffer()
