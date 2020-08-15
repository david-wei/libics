import time

from libics.driver.emulator import ItfEmulator
from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal
from libics.driver.terminal.gpib import ItfGpib, GPIB_MODE


###############################################################################


class PrologixGpib(ItfEmulator, ItfGpib):

    def __init__(self):
        super().__init__()
        self._is_set_up = False
        self._is_connected = False

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
        self._is_set_up = True

    def shutdown(self):
        if self.is_set_up():
            self.interface.shutdown()
        self._is_set_up = False

    def is_set_up(self):
        return self.interface.is_set_up() and self._is_set_up

    def connect(self, id):
        if not self.is_set_up():
            raise RuntimeError("device not set up")
        self.interface.connect(self.identifier)
        self.interface.register(self.identifier, self)
        # Disable in-device cfg saving
        self.send("++savecfg 0")
        # Set GPIB mode
        _MODE = {self.GPIB_MODE.CONTROLLER: 1, self.GPIB_MODE.DEVICE: 0}
        self.send("++mode {:d}".format(_MODE[self.cfg.gpib_mode]))
        # Enable EOI assertion
        self.send("++eoi 1")
        # Disable automatic read after write mode
        self.send("++auto 0")
        # Disable auto-appending termchars
        self.send("++eos 3")
        # Set GPIB address
        self.send("++addr {:d}".format(self.cfg.gpib_address))
        # Set read timeout
        self.send("++read_tmo_ms {:d}"
                  .format(round(1000 * self.cfg.recv_timeout)))
        time.sleep(2)
        self._is_connected = True

    def close(self):
        # Set local
        self.send("++loc")
        self.interface.deregister(id=self.identifier)
        self.interface.close()
        self._is_connected = False

    def is_connected(self):
        return self.interface.is_connected() and self._is_connected

    # ++++++++++++++++++++++++++++++++++++++++
    # ItfTerminal methods
    # ++++++++++++++++++++++++++++++++++++++++

    def send(self, msg):
        return self.interface.send(msg)

    def recv(self):
        if self.mode == GPIB_MODE.CONTROLLER:
            self.send("++read")
        return self.interface.recv()

    def flush_out(self):
        return self.interface.flush_out()

    def flush_in(self):
        return self.interface.flush_in()
