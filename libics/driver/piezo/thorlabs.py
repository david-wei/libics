import time

from libics.core.env import logging
from libics.driver.piezo import PiezoActuator
from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal


###############################################################################
# Device
###############################################################################


class ThorlabsMDT69X(PiezoActuator):

    LOGGER = logging.get_logger("libics.driver.piezo.thorlabs.ThorlabsMDT69X")

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    def setup(self):
        """
        Raises
        ------
        RuntimeError
            If interface is not set.
        """
        if not isinstance(self.interface, ItfTerminal):
            err_msg = "invalid interface"
            self.last_status = STATUS(
                state=STATUS.ERROR, err_type=STATUS.ERR_CONNECTION, msg=err_msg
            )
            raise RuntimeError(err_msg)
        self.interface.setup()

    def shutdown(self):
        if self.is_connected():
            self.close()
        if self.is_set_up():
            self.interface.shutdown()

    def is_set_up(self):
        return self.interface.is_set_up()

    def connect(self):
        if not self.is_set_up():
            raise RuntimeError("device not set up")
        self.interface.connect(self.identifier)
        self.interface.register(self.identifier, self)
        self._turn_off_echo_mode()
        self.p.read_all()

    def close(self):
        self.interface.deregister(id=self.identifier)
        self.interface.close()

    def is_connected(self):
        return self.interface.is_connected()

    # ++++++++++++++++++++++++++++++++++++++++
    # Helper methods
    # ++++++++++++++++++++++++++++++++++++++++

    def _turn_off_echo_mode(self):
        self.interface.send("E")
        time.sleep(0.5)
        s_recv = self.interface.recv().lstrip("\n\r*[e ").rstrip("\n\r] ")
        if s_recv == "Echo On":
            self.interface.send("E")
            time.sleep(0.5)
            self.interface.recv()

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_voltage(self):
        MAP = {0: "X", 1: "Y", 2: "Z"}
        value = float(self.interface.query("{:s}R?".format(MAP[self.channel])))
        self.p.voltage = value
        return value

    def write_voltage(self, value):
        MAP = {0: "X", 1: "Y", 2: "Z"}
        self.interface.send("{:s}V{:.3f}".format(MAP[self.channel], value))
        self.p.voltage = value

    def read_voltage_min(self):
        MAP = {0: "X", 1: "Y", 2: "Z"}
        value = float(self.interface.query("{:s}L?".format(MAP[self.channel])))
        self.p.voltage_min = value
        return value

    def write_voltage_min(self, value):
        MAP = {0: "X", 1: "Y", 2: "Z"}
        self.interface.send("{:s}L{:.3f}".format(MAP[self.channel], value))
        self.p.voltage_min = value

    def read_voltage_max(self):
        MAP = {0: "X", 1: "Y", 2: "Z"}
        value = float(self.interface.query("{:s}H?".format(MAP[self.channel])))
        self.p.voltage_max = value
        return value

    def write_voltage_max(self, value):
        MAP = {0: "X", 1: "Y", 2: "Z"}
        self.interface.send("{:s}H{:.3f}".format(MAP[self.channel], value))
        self.p.voltage_max = value
