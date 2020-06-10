from libics.driver.interface import ItfBase


###############################################################################


class ItfEmulator(ItfBase):

    def __init__(self):
        super().__init__()
        raise NotImplementedError
