import time

from libics.driver.emulator import ItfEmulatorBase
from libics.driver.terminal import gpib


###############################################################################


class ItfEmulatorPrologixGpib(ItfEmulatorBase, gpib.ItfGpib):

    def __init__(self, cfg):
        super().__init__(cfg)

    def connect(self):
        super().connect()
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

    def close(self):
        # Set local
        self.send("++loc")
        super().close()

    def recv(self):
        if self.cfg.gpib_mode == self.GPIB_MODE.CONTROLLER:
            self.send("++read")
        return super().recv()
