# System Imports
import glob
import serial
import sys


###############################################################################


def _list_serial_ports():
    """
    Lists serial port names.

    Returns
    -------
    result : list(str)
        A list of the serial ports available on the system.

    Raises
    ------
    EnvironmentError
        If platform is not supported or unknown.
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
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


###############################################################################


class MDT693(object):

    """
    RS232 (serial port) driver for Thorlabs Piezo Driver MDT693A.
    """

    def __init__(self,
                 port="COM1", baudrate=115200, parity=serial.PARITY_NONE,
                 stopbits=serial.STOPBITS_ONE,
                 read_timeout=1.0, write_timeout=1.0):
        self._serial = serial.Serial(
            port=port, baudrate=baudrate, parity=parity, stopbits=stopbits,
            timeout=read_timeout, write_timeout=write_timeout
        )
        self._voltage_limit = 75

    def open_serial(self):
        if not self._serial.is_open:
            self._serial.open()
        self._voltage_limit = self.get_voltage_limit()
        return self._serial.is_open

    def close_serial(self):
        if self._serial.is_open:
            self._serial.close()
        return not self._serial.is_open

    @staticmethod
    def _get_channel_code(channel):
        if channel is None or channel == "all":
            return "A"
        elif channel == 0 or channel == "x":
            return "X"
        elif channel == 1 or channel == "y":
            return "Y"
        elif channel == 2 or channel == "z":
            return "Z"

    def _send(self, cmd_str, voltage=None):
        cmd_b = bytes(cmd_str, "ascii")
        if voltage is not None:
            voltage = int(round(
                float(voltage) / self._voltage_limit * 65536
            ))
            cmd_b += voltage.to_bytes(2, byteorder="big")
        self._serial.write(cmd_b)

    def _receive(self, byte_count=2, raw=False):
        ret = self._serial.read(size=byte_count)
        ret = int.from_bytes(ret, byteorder="big")
        if not raw:
            ret = float(ret) / 65536 * self._voltage_limit
        return ret

    def set_voltage(self, voltage, channel=None):
        self._send(MDT693._get_channel_code(channel) + "V", voltage)

    def get_voltage(self, channel=None):
        return self._receive(MDT693._get_channel_code(channel) + "R?")

    def set_voltage_range(self, min_volt=None, max_volt=None, channel=None):
        if min_volt is not None:
            self._send(MDT693._get_channel_code(channel) + "L", min_volt)
        if max_volt is not None:
            self._send(MDT693._get_channel_code(channel) + "H", max_volt)

    def get_voltage_range(self, channel=None):
        """
        Returns the voltage range [min, max] or [[min0, max0], ...].
        """
        if channel is None:
            data = []
            for ch in range(2):
                data.append(self.get_voltage_range(channel=ch))
        else:
            data = []
            self._send(MDT693._get_channel_code(channel) + "L?")
            data.append(self._receive(byte_count=2, raw=False))
            self._send(MDT693._get_channel_code(channel) + "H?")
            data.append(self._receive(byte_count=2, raw=False))
        return data

    def get_voltage_limit(self):
        self._send("%")
        ret = self._receive(byte_count=1, raw=True)
        if ret == 0:
            return 75
        elif ret == 1:
            return 100
        elif ret == 2:
            return 150


###############################################################################


if __name__ == "__main__":

    # Test settings
    _serial_ports = _list_serial_ports()
    print("Serial ports:", _serial_ports)
    port = _serial_ports[-1]
    baudrate = 115200

    # Setup test
    piezo = MDT693(port=port, baudrate=baudrate)
    piezo.open_serial()
    print("Piezo voltage limit:", piezo.get_voltage_limit())
    piezo.close_serial()
