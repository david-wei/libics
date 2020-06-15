import time

from libics.core.env import logging
from libics.core.util import misc
from libics.driver.piezo import Picomotor
from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal


###############################################################################
# Device
###############################################################################

class Newport8742(Picomotor):

    """
    Configuration
    -------------
    subaddress : `str`
        Device address (0-255) in 8742-internal LAN.

    Properties
    ----------
    device_name : `str`
        Manufacturer device ID.
    """

    LOGGER = logging.get_logger("libics.driver.piezo.newport.Newport8742")

    def __init__(self):
        super().__init__()
        self.channel = 1
        self.subaddress = None
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
        self.interface.connect()
        self.interface.register(self.identifier, self)
        self._turn_off_echo_mode()
        self.p.read_all()

    def close(self):
        self.interface.deregister(id=self.identifier)
        self.interface.close()

    def is_connected(self):
        return self.interface.is_connected()

    # ++++++++++++++++++++++++++++++++++++++++
    # Picomotor methods
    # ++++++++++++++++++++++++++++++++++++++++

    def abort_motion(self):
        self._itf_send("AB", use_channel=True)

    def home_position(self):
        self._itf_send("DH", use_channel=True)

    def move_relative(self, steps):
        self._itf_send("PR{:.0f}".format(steps), use_channel=True)
        _repetitions = 30
        _wait = max(0.1, steps / self.p.velocity)
        for _i in range(_repetitions):
            time.sleep(_wait)
            self._itf_send("MD?", use_channel=True)
            if int(self._itf_recv()) == 1:
                break
            if _i >= _repetitions - 1:
                err_msg = "move_relative timeout"
                self.last_status = STATUS(
                    state=STATUS.ERROR, err_type=STATUS.ERR_DEVICE, msg=err_msg
                )
                raise RuntimeError(err_msg)

    # ++++++++++++++++++++++++++++++++++++++++
    # Newport8742 methods
    # ++++++++++++++++++++++++++++++++++++++++

    def discover_subaddresses(self):
        """
        Scans for slave devices in device-internal LAN.

        Returns
        -------
        subaddresses : `list(str)`
            Available slave device subaddresses.

        Raises
        ------
        ValueError
            If scan timed out or slave devices have address conflicts.
        """
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
        if bf[0]:
            raise RuntimeError("Address conflict")
        subaddresses = []
        for i in bf[1:]:
            if i:
                subaddresses.append(i)
        return subaddresses

    def read_error(self):
        value = self._itf_query("TB?", use_channel=False)
        return value

    # ++++++++++++++++++++++++++++++++++++++++
    # Helper methods
    # ++++++++++++++++++++++++++++++++++++++++

    def _itf_send(self, msg, use_channel=False):
        prefix = ""
        if self.subaddress is not None:
            prefix = prefix + "{:s}>".format(self.subaddress)
        if use_channel:
            prefix = prefix + "{:d}".format(self.channel)
        self.interface.send(prefix + msg)

    def _itf_recv(self):
        msg = self._interface.recv()
        msg = msg.split(">")[-1]
        return msg

    def _itf_query(self, msg, use_channel=False):
        self._itf_send(msg, use_channel=use_channel)
        return self._itf_recv()

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_position(self):
        value = float(self._itf_query("PR?", use_channel=True))
        self.p.position = value
        return value

    def write_position(self, value):
        self.LOGGER.warning("cannot write position")

    def read_acceleration(self):
        value = float(self._itf_query("AC?", use_channel=True))
        self.p.acceleration = value
        return value

    def write_acceleration(self, value):
        self._itf_send("AC{:.0f}".format(value), use_channel=True)
        self.p.acceleration = value

    def read_velocity(self):
        value = float(self._itf_query("VA?", use_channel=True))
        self.p.velocity = value
        return value

    def write_velocity(self, value):
        self._itf_send("VA{:.0f}".format(value), use_channel=True)
        self.p.velocity = value

    def read_device_name(self):
        value = self._itf_query("*IDN?", use_channel=False)
        value = misc.extract(value, r"New_Focus 8742 v.* (\d+)")
        self.p.device_name = value
        return value

    def write_device_name(self, value):
        if value != self.p.device_name:
            self.LOGGER.warning("cannot write device_name")
