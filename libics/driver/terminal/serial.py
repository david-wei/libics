import serial

from libics.driver.interface import ItfTerminal





@InheritMap(map_key=("libics", "TxtSerialCfg"))
class TxtSerialCfg(TxtCfgBase):

    """
    ProtocolCfgBase -> TxtCfgBase -> TxtSerialCfg.

    Parameters
    ----------
    baudrate : int
        Baud rate in bits per second.
    bytesize : 5, 6, 7, 8
        Number of data bits.
    parity : "none", "even", "odd", "mark", "space"
        Parity checking.
    stopbits : 1, 1.5, 2
        Number of stop bits.
    """

    def __init__(self, baudrate=115200, bytesize=8, parity="none",
                 stopbits=1,
                 cls_name="TxtSerialCfg", ll_obj=None, **kwargs):
        if "interface" not in kwargs.keys():
            kwargs["interface"] = ITF_TXT.SERIAL
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits

    def get_hl_cfg(self):
        return self






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

    def __init__(self):
        super().__init__()
        self._serial = None

    def setup(self):
        self._serial = serial.Serial(
            port=self.address,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.cfg.stopbits,
            timeout=self.recv_timeout,
            write_timeout=self.send_timeout
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
