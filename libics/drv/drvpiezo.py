# System Imports
import abc
import time

# Package Imports
from libics.drv import drv


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
        self._interface.send("{:s}VR?".format(str(self.cfg.channel)))
        voltage = float(self._interface.recv())
        self.interface_access.release()
        return voltage

    # ++++ Write/read methods +++++++++++

    def _write_limit_min(self, value):
        self._interface.send(
            "{:s}L{:.3f}".format(str(self.cfg.channel), value)
        )

    def _read_limit_min(self):
        self._interface.send("{:s}L?")
        return float(self._interface.recv())

    def _write_limit_max(self, value):
        self._interface.send(
            "{:s}H{:.3f}".format(str(self.cfg.channel), value)
        )

    def _read_limit_max(self):
        self._interface.send("{:s}H?")
        return float(self._interface.recv())

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

    def _turn_off_echo_mode(self):
        self._interface.send("E")
        time.sleep(0.1)
        s_recv = self._interface.recv().lstrip("\n\r\*[e ").rstrip("\n\r] ")
        if s_recv == "Echo On":
            self._interface.send("E")
            time.sleep(0.1)
            self._interface.recv()
