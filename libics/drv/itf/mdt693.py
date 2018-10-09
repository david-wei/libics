# System Imports
import glob
import serial
import sys
import time


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


class MDT693A(object):

    """
    RS232 (serial port) driver for Thorlabs Piezo Driver MDT693A.
    """

    MODE_RAW = 0
    MODE_VOLTAGE = 1
    MODE_CODE = 2

    def __init__(self,
                 port="COM1", baudrate=115200, parity=serial.PARITY_NONE,
                 stopbits=serial.STOPBITS_ONE,
                 read_timeout=1.0, write_timeout=1.0,
                 termchar="\r\n", hw_proc_delay=0.05,
                 debug=False):
        self._debug = debug
        self._serial = serial.Serial(
            port=port, baudrate=baudrate, parity=parity, stopbits=stopbits,
            timeout=read_timeout, write_timeout=write_timeout
        )
        self._termchar = termchar
        self._hw_proc_delay = hw_proc_delay
        self._voltage_limit = 75

    def open_serial(self):
        if not self._serial.is_open:
            self._serial.open()
        self._turn_off_echo_mode()
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
        if voltage is not None:
            cmd_str += "{:.2f}".format(float(voltage))
        cmd_str += self._termchar
        cmd_b = cmd_str.encode("ascii")
        self._serial.write(cmd_b)
        self._serial.flush()
        if self._debug:
            print("send:", cmd_b)

    def _receive(self):
        ret_b = self._serial.readline()
        ret_str = ret_b.decode("ascii")
        ret_str = ret_str.lstrip("\n\r\*[ \x00").rstrip("\n\r] \x00")
        if self._debug:
            print("recv:", ret_b)
        return ret_str

    def _turn_off_echo_mode(self):
        self._send("E")
        # wait for MDT693 to change settings
        time.sleep(10 * self._hw_proc_delay)
        ret_b = self._serial.readline()
        if self._debug:
            print("recv:", ret_b)
        ret_str = ret_b.decode("ascii").lstrip("\n\r\*[e ").rstrip("\n\r] ")
        if ret_str == "Echo On":
            self._send("E")
            time.sleep(10 * self._hw_proc_delay)
            self._receive()

    def set_voltage(self, voltage, channel=None):
        self._send(MDT693A._get_channel_code(channel) + "V", voltage)
        time.sleep(self._hw_proc_delay)

    def get_voltage(self, channel=0):
        """
        Reads and returns the voltage of the given channel.

        Raises
        ------
        RuntimeError
            If `channel` is invalid.
        """
        if channel is None:
            raise RuntimeError(
                "libics.drv.itf.mdt693.MDT693.get_voltage: invalid channel"
            )
        self._send(MDT693A._get_channel_code(channel) + "R?")
        ret_str = self._receive()
        voltage = float(ret_str)
        return voltage

    def set_voltage_range(self, min_volt=None, max_volt=None, channel=None):
        if min_volt is not None:
            self._send(MDT693A._get_channel_code(channel) + "L", min_volt)
            time.sleep(self._hw_proc_delay)
        if max_volt is not None:
            self._send(MDT693A._get_channel_code(channel) + "H", max_volt)
            time.sleep(self._hw_proc_delay)

    def get_voltage_range(self, channel=0):
        """
        Returns the voltage range (min, max).

        Raises
        ------
        RuntimeError
            If `channel` is invalid.
        """
        if channel is None:
            raise RuntimeError(
                "libics.drv.itf.mdt693.MDT693.get_voltage: invalid channel"
            )
        else:
            self._send(MDT693A._get_channel_code(channel) + "L?")
            ret_min = float(self._receive())
            self._send(MDT693A._get_channel_code(channel) + "H?")
            ret_max = float(self._receive())
        return ret_min, ret_max

    def get_voltage_limit(self):
        self._send("%")
        ret_str = self._receive()
        voltage_limit = float(ret_str)
        return voltage_limit


###############################################################################


class MDT693B(object):

    """
    RS232 (serial port) driver for Thorlabs Piezo Driver MDT693B.
    """

    MODE_RAW = 0
    MODE_VOLTAGE = 1
    MODE_CODE = 2

    def __init__(self,
                 port="COM1", baudrate=115200, parity=serial.PARITY_NONE,
                 stopbits=serial.STOPBITS_ONE,
                 read_timeout=1.0, write_timeout=1.0,
                 termchar="\r\n", hw_proc_delay=0.05,
                 debug=False):
        self._debug = debug
        self._serial = serial.Serial(
            port=port, baudrate=baudrate, parity=parity, stopbits=stopbits,
            timeout=read_timeout, write_timeout=write_timeout
        )
        self._termchar = termchar
        self._hw_proc_delay = hw_proc_delay
        self._voltage_limit = 75

    def open_serial(self):
        if not self._serial.is_open:
            self._serial.open()
        self._turn_off_echo_mode()
        self._voltage_limit = self.get_voltage_limit()
        return self._serial.is_open

    def close_serial(self):
        if self._serial.is_open:
            self._serial.close()
        return not self._serial.is_open

    @staticmethod
    def _get_channel_code(channel):
        if channel is None or channel == "all":
            return "all"
        elif channel == 0 or channel == "x":
            return "x"
        elif channel == 1 or channel == "y":
            return "y"
        elif channel == 2 or channel == "z":
            return "Z"

    def _send(self, cmd_str, voltage=None):
        if voltage is not None:
            cmd_str += str(voltage)
        cmd_str += self._termchar
        cmd_b = cmd_str.encode("ascii")
        self._serial.write(cmd_b)
        self._serial.flush()
        if self._debug:
            print("send:", cmd_b)

    def _receive(self):
        ret_b = self._serial.readline()
        ret_str = ret_b.decode("ascii")
        ret_str = ret_str.lstrip("\n\r\*[ \x00").rstrip("\n\r] \x00")
        if self._debug:
            print("recv:", ret_b)
        return ret_str

    def _turn_off_echo_mode(self):
        self._send("echo=0")
        time.sleep(10 * self._hw_proc_delay)

    def set_voltage(self, voltage, channel=None):
        self._send(MDT693B._get_channel_code(channel) + "voltage=", voltage)
        time.sleep(self._hw_proc_delay)

    def get_voltage(self, channel=0):
        """
        Reads and returns the voltage of the given channel.

        Raises
        ------
        RuntimeError
            If `channel` is invalid.
        """
        if channel is None:
            raise RuntimeError(
                "libics.drv.itf.mdt693.MDT693.get_voltage: invalid channel"
            )
        self._send(MDT693B._get_channel_code(channel) + "voltage?")
        ret_str = self._receive()
        voltage = float(ret_str)
        return voltage

    def set_voltage_range(self, min_volt=None, max_volt=None, channel=None):
        if channel is None:
            for ch in ["x", "y", "z"]:
                self.set_voltage_range(min_volt=min_volt, max_volt=max_volt,
                                       channel=ch)
        if min_volt is not None:
            self._send(MDT693B._get_channel_code(channel) + "min=", min_volt)
            time.sleep(self._hw_proc_delay)
        if max_volt is not None:
            self._send(MDT693B._get_channel_code(channel) + "max=", max_volt)
            time.sleep(self._hw_proc_delay)

    def get_voltage_range(self, channel=0):
        """
        Returns the voltage range (min, max).

        Raises
        ------
        RuntimeError
            If `channel` is invalid.
        """
        if channel is None:
            raise RuntimeError(
                "libics.drv.itf.mdt693.MDT693.get_voltage: invalid channel"
            )
        else:
            self._send(MDT693B._get_channel_code(channel) + "min?")
            ret_min = float(self._receive())
            self._send(MDT693B._get_channel_code(channel) + "max?")
            ret_max = float(self._receive())
        return ret_min, ret_max

    def get_voltage_limit(self):
        self._send("vlimit?")
        ret_lim = int(self._receive())
        ret_lim = None
        if ret_lim == 0:
            voltage_limit = 75.0
        elif ret_lim == 1:
            voltage_limit = 100.0
        elif ret_lim == 2:
            voltage_limit = 150.0
        return voltage_limit


###############################################################################


if __name__ == "__main__":

    # Test settings
    _serial_ports = _list_serial_ports()
    print("Serial ports:", _serial_ports)
    index = input("Choose serial port: ")
    try:
        index = int(index)
    except ValueError:
        index = 0
    index = index % len(_serial_ports)
    port = _serial_ports[index]
    baudrate = 115200
    debug = True

    # Setup test
    piezo = MDT693A(port=port, baudrate=baudrate, debug=debug)
    piezo.open_serial()
    print("Piezo voltage limit [V]:", piezo.get_voltage_limit())
    print("Piezo voltage [V]:", piezo.get_voltage())
    piezo.close_serial()
