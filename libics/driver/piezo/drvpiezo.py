# System Imports
import abc
import time
import numpy as np

# Package Imports
from libics.drv import drv
from libics.util import misc


###############################################################################


def get_piezo_drv(cfg):
    if cfg.model == drv.DRV_MODEL.THORLABS_MDT69XA:
        return ThorlabsMDT69XA(cfg)


class PiezoDrvBase(drv.DrvBase):

    """
    Piezo driver API.
    """

    def __init__(self, cfg):
        assert(isinstance(cfg, drv.PiezoCfg))
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def write_voltage(self, voltage):
        pass

    @abc.abstractmethod
    def read_voltage(self):
        pass

    # ++++ Write/read methods +++++++++++

    def _write_limit_min(self, value):
        pass

    def _read_limit_min(self):
        pass

    def _write_limit_max(self, value):
        pass

    def _read_limit_max(self):
        pass

    def _write_displacement(self, value):
        pass

    def _read_displacement(self):
        pass

    def _write_channel(self, value):
        pass

    def _read_channel(self):
        pass

    def _write_feedback_mode(self, value):
        pass

    def _read_feedback_mode(self):
        pass


###############################################################################


class ThorlabsMDT69XA(PiezoDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        ThorlabsMDT69XA._assert_channel(cfg.channel.val)
        self.__channel = cfg.channel.val

    def connect(self):
        super().connect()
        self._turn_off_echo_mode()

    def write_voltage(self, voltage):
        self.interface_access.acquire()
        self._interface.send(
            "{:s}V{:.3f}".format(str(self.cfg.channel), voltage)
        )
        time.sleep(0.05)
        self.interface_access.release()

    def read_voltage(self):
        self.interface_access.acquire()
        self._interface.send("{:s}R?".format(str(self.cfg.channel)))
        voltage = float(ThorlabsMDT69XA._strip_recv(self._interface.recv()))
        self.interface_access.release()
        return voltage

    # ++++ Write/read methods +++++++++++

    def _write_limit_min(self, value):
        self._interface.send(
            "{:s}L{:.3f}".format(str(self.cfg.channel), value)
        )
        time.sleep(0.05)

    def _read_limit_min(self):
        self._interface.send("{:s}L?".format(str(self.cfg.channel)))
        return float(ThorlabsMDT69XA._strip_recv(self._interface.recv()))

    def _write_limit_max(self, value):
        self._interface.send(
            "{:s}H{:.3f}".format(str(self.cfg.channel), value)
        )
        time.sleep(0.05)

    def _read_limit_max(self):
        self._interface.send("{:s}H?".format(str(self.cfg.channel)))
        return float(ThorlabsMDT69XA._strip_recv(self._interface.recv()))

    def _write_channel(self, value):
        self.__channel = ThorlabsMDT69XA._assert_channel(value)

    def _read_channel(self):
        return self.__channel

    def _write_feedback_mode(self, value):
        ThorlabsMDT69XA._assert_feedback_mode(value)

    def _read_feedback_mode(self):
        return drv.DRV_PIEZO.FEEDBACK_MODE.OPEN_LOOP

    # ++++ Helper methods +++++++++++++++

    @staticmethod
    def _assert_channel(value):
        if value not in ["x", "y", "z", "X", "Y", "Z"]:
            raise ValueError("invalid channel: {:s}".format(str(value)))
        return value

    @staticmethod
    def _assert_feedback_mode(value):
        if value != drv.DRV_PIEZO.FEEDBACK_MODE.OPEN_LOOP:
            raise ValueError("invalid feedback mode: {:s}".format(str(value)))
        return value

    @staticmethod
    def _strip_recv(value):
        return value.lstrip("\n\r*[ \x00").rstrip("\n\r] \x00")

    def _turn_off_echo_mode(self):
        self._interface.send("E")
        time.sleep(0.5)
        s_recv = self._interface.recv().lstrip("\n\r*[e ").rstrip("\n\r] ")
        if s_recv == "Echo On":
            self._interface.send("E")
            time.sleep(0.5)
            self._interface.recv()


###############################################################################


def get_pico_drv(cfg):
    if cfg.model == drv.DRV_MODEL.NEWPORT_8742:
        return Newport8742(cfg)


class PicoDrvBase(drv.DrvBase):

    """
    Picomotor driver API.
    """

    def __init__(self, cfg):
        assert(isinstance(cfg, drv.PicoCfg))
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def abort_motion(self):
        pass

    @abc.abstractmethod
    def zero_position(self):
        pass

    @abc.abstractmethod
    def move_relative(self, steps):
        pass

    # ++++ Write/read methods +++++++++++

    def _write_feedback_mode(self, value):
        pass

    def _read_feedback_mode(self):
        pass


###############################################################################


class Newport8742(PicoDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg)
        Newport8742._assert_channel(cfg.channel.val)
        self._write_channel(cfg.channel.val)

    def connect(self):
        super().connect()
        self._assert_identifier()

    def read_position(self):
        self.interface_access.acquire()
        self._itf_send("PR?", use_channel=True)
        position = int(self._itf_recv())
        self.interface_access.release()
        return position

    def abort_motion(self):
        self.interface_access.acquire()
        self._itf_send("AB", use_channel=True)
        self.interface_access.release()

    def zero_position(self):
        self.interface_access.acquire()
        self._itf_send("DH", use_channel=True)
        self.interface_access.release()

    def move_relative(self, steps):
        """
        Raises
        ------
        ValueError
            If timeout.
        """
        self.interface_access.acquire()
        self._itf_send("PR{:.0f}".format(steps), use_channel=True)
        _repetitions = 30
        _wait = max(0.1, steps / self.cfg.velocity.val)
        for _i in range(_repetitions):
            time.sleep(_wait)
            self._itf_send("MD?", use_channel=True)
            if int(self._itf_recv()) == 1:
                break
            if _i >= _repetitions - 1:
                raise RuntimeError("Scan timeout")
        self.interface_access.release()

    def scan_slave_devices(self):
        """
        Returns
        -------
        addr : list(int)
            Available slave device addresses.

        Raises
        ------
        ValueError
            If scan timed out or slave devices have address conflicts.
        """
        self.interface_access.acquire()
        self._itf_send("SC", use_channel=False)
        _repetitions = 30
        for _i in range(_repetitions):
            time.sleep(0.1)
            self._itf_send("SD?", use_channel=False)
            if int(self._itf_recv()) == 1:
                break
            if _i >= _repetitions - 1:
                raise RuntimeError("Scan timeout")
        self._itf_send("SC?", use_channel=False)
        bf = list(reversed(misc.cv_bitfield(int(self._itf_recv()))))
        self.interface_access.release()
        if bf[0]:
            raise RuntimeError("Address conflict")
        addr = []
        for i in bf[1:]:
            if i:
                addr.append(i)
        return addr

    def read_error(self):
        self.interface_access.acquire()
        self._itf_send("TB?", use_channel=False)
        _recv = self._itf_recv()
        self.interface_access.release()
        return _recv

    # ++++ Write/read methods +++++++++++

    def _write_channel(self, value):
        self.__channel = Newport8742._assert_channel(value)

    def _read_channel(self):
        return self.__channel

    def _write_acceleration(self, value):
        self._itf_send("AC{:.0f}".format(value), use_channel=True)
        time.sleep(0.01)

    def _read_acceleration(self):
        self._itf_send("AC?", use_channel=True)
        return int(self._itf_recv())

    def _write_velocity(self, value):
        self._itf_send("VA{:.0f}".format(value), use_channel=True)
        time.sleep(0.01)

    def _read_velocity(self):
        self._itf_send("VA?", use_channel=True)
        return int(self._itf_recv())

    def _write_feedback_mode(self, value):
        Newport8742._assert_feedback_mode(value)

    def _read_feedback_mode(self):
        return drv.DRV_PICO.FEEDBACK_MODE.OPEN_LOOP

    # ++++ Helper methods +++++++++++++++

    def _assert_identifier(self):
        self.interface_access.acquire()
        self._itf_send("*IDN?", use_channel=False)
        _id = misc.extract(self._itf_recv(), r"New_Focus 8742 v.* (\d+)")
        self.interface_access.release()
        if _id != self.cfg.identifier:
            raise ValueError("Wrong device ID ({:s})".format(_id))

    def _itf_send(self, msg, use_channel=False):
        prefix = ""
        if len(self.__channel) == 2:
            prefix = prefix + "{:.0f}>".format(self.__channel[0])
        if use_channel:
            prefix = prefix + "{:.0f}".format(self.__channel[-1])
        self._interface.send(prefix + msg)

    def _itf_recv(self):
        msg = self._interface.recv()
        msg = msg.split(">")[-1]
        return self._strip_recv(msg)

    @staticmethod
    def _assert_channel(value):
        value = misc.assume_iter(value)
        if not (
            (len(value) == 1 or len(value) == 2)
            and np.all([isinstance(item, int) for item in value])
        ):
            raise ValueError("invalid channel: {:s}".format(str(value)))
        return value

    @staticmethod
    def _assert_feedback_mode(value):
        if value != drv.DRV_PICO.FEEDBACK_MODE.OPEN_LOOP:
            raise ValueError("invalid feedback mode: {:s}".format(str(value)))
        return value

    @staticmethod
    def _strip_recv(value):
        return value.lstrip("\n\r*[ \x00").rstrip("\n\r] \x00")
