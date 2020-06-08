from libics.core.env import logging
from libics.driver.terminal import ItfTerminal


###############################################################################


class MODE:

    CONTROLLER = "CONTROLLER"
    DEVICE = "DEVICE"


###############################################################################


class ItfGpib(ItfTerminal):

    """
    Parameters
    ----------
    mode : `MODE`
        GPIB mode (controller or device).
    address : `int`
        GPIB address (1-32).
    """

    MODE = MODE.CONTROLLER
    ADDRESS = 1

    LOGGER = logging.get_logger("libics.driver.terminal.ethernet.ItfGpib")

    def __init__(self):
        super().__init__()
        self._itf = None
        self.mode = self.MODE

    def configure(self, itf=None, **cfg):
        if itf is None:
            raise RuntimeError("no GPIB interface (itf) specified")
        self._itf = itf
        super().configure(**cfg)
