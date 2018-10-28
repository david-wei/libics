# System Imports
import abc
import time

# Package Imports
from libics.cfg import CFG_MSG_TYPE
from libics.drv import drv, itf


###############################################################################


def get_piezo_drv(cfg):
    if cfg.model == drv.DRV_MODEL.THORLABS_MDT69XA:
        return ThorlabsMDT69XA(cfg)


class PiezoDrvBase(abc.ABC):

    def __init__(self, cfg=None):
        self.cfg = cfg

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def write(self, msg):
        pass

    @abc.abstractmethod
    def read(self, msg):
        pass

    def process(self):
        """
        Processes the message queue in the configuration object.
        """
        msg = self.cfg._pop_msg()
        while (msg is not None):
            if (msg.msg_type == CFG_MSG_TYPE.WRITE or
                    msg.msg_type == CFG_MSG_TYPE.VALIDATE):
                self.write(msg)
            if (msg.msg_type == CFG_MSG_TYPE.READ or
                    msg.msg_type == CFG_MSG_TYPE.VALIDATE):
                self.read(msg)
            msg = self.cfg._pop_msg()

    @abc.abstractmethod
    def set_voltage(self, voltage):
        pass

    @abc.abstractmethod
    def get_voltage(self):
        pass


###############################################################################


class ThorlabsMDT69XA(PiezoDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self._interface = None
        self.setup()

    def setup(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg
        self._interface = itf.get_itf(cfg.interface)
        self._interface.setup()

    def shutdown(self):
        self._interface.shutdown()

    def connect(self):
        self._interface.connect()
        self._turn_off_echo_mode()

    def close(self):
        self._interface.close()

    def write(self, msg):
        pass

    def read(self, msg):
        pass

    def set_voltage(self, voltage):
        self._interface.send(
            "{:s}V{:.3f}".format(str(self.cfg.channel), voltage)
        )
        time.sleep(0.05)

    def get_voltage(self):
        self._interface.send("{:s}VR?".format(str(self.cfg.channel)))
        s_data = self._interface.recv()
        voltage = float(s_data)
        return voltage

    # ++++ Helper methods +++++++++++++++

    def _turn_off_echo_mode(self):
        self._interface.send("E")
        time.sleep(0.1)
        s_recv = self._interface.recv().lstrip("\n\r\*[e ").rstrip("\n\r] ")
        if s_recv == "Echo On":
            self._interface.send("E")
            time.sleep(0.1)
            self._interface.recv()
