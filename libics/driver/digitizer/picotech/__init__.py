import ctypes

from libics.driver.interface import ItfBase
from libics.driver.digitizer import Thermocouple

tc08api = ctypes.windll.LoadLibrary("usbtc08.dll")


###############################################################################


class ItfPicotech(ItfBase):

    def __init__(self):
        super().__init__()
        raise NotImplementedError


class PicotechTC08(Thermocouple):

    def __init__(self):
        super().__init__()
        raise NotImplementedError