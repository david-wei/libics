from libics.core.env import logging
from libics.driver.lightsource import LightSource
from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal


###############################################################################
# Device
###############################################################################


class IpgYLR(LightSource):

    """
    Properties
    ----------
    current : `float`
        Relative light source current with respect to the maximal current.
    device_name : `str`
        Manufacturer device ID.
    """

    LOGGER = logging.get_logger("libics.driver.lightsource.ipg.IpgYLR")

    def __init__(self):
        super().__init__()
        self.properties.set_properties(self._get_default_properties_dict(
            "device_name"
        ))

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
        if self.is_set_up():
            self.interface.shutdown()

    def is_set_up(self):
        return self.interface.is_set_up()

    def connect(self):
        self.interface.connect(self.identifier)
        self.interface.register(self.identifier, self)
        self.p.read_all()

    def close(self):
        self.interface.deregister(id=self.identifier)
        self.interface.close()

    def is_connected(self):
        return self.interface.is_connected()

    # ++++++++++++++++++++++++++++++++++++++++
    # LightSource methods
    # ++++++++++++++++++++++++++++++++++++++++

    def run(self):
        self.interface.send("EMON")

    def stop(self):
        self.interface.send("EMOFF")

    def is_running(self):
        return self.read_power() != -1

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_current(self):
        value = float(self.interface.query("RCS").split(":")[-1].strip(" "))
        self.p.current = value
        return value

    def write_current(self, value):
        self.interface.query("SDC {:.2f}".format(value))
        self.p.current = value

    def read_power(self):
        value = self.interface.query("ROP").split(":")[-1].strip(" ")
        if value.upper() == "OFF":
            value = -1
        elif value.upper() == "LOW":
            value = 0
        self.p.power = value
        return value

    def write_power(self, value):
        self.LOGGER.warning("cannot write power")

    def read_temperature(self):
        value = float(self.interface.query("RCT").split(":")[-1].strip(" "))
        self.p.temperature = value
        return value

    def write_temperature(self, value):
        self.LOGGER.warning("cannot write temperature")

    def read_emission_time(self):
        value = float(self.interface.query("RET").split(":")[-1].strip(" "))
        value *= 60
        self.p.emission_time = value
        return value

    def write_emission_time(self, value):
        self.LOGGER.warning("cannot write emission_time")

    def read_device_name(self):
        value = self.interface.query("RSN").split(":")[-1].strip(" ")
        self.p.device_name = value
        return value

    def write_device_name(self, value):
        if value != self.p.device_name:
            self.LOGGER.warning("cannot write device_name")
