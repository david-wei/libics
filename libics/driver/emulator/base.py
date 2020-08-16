from libics.driver.interface import ItfBase


###############################################################################


class ItfEmulator(ItfBase):

    """
    Relays interface calls to the object's own interface.
    """

    # ++++++++++++++++++++++++++++++++++++++++
    # Interface methods
    # ++++++++++++++++++++++++++++++++++++++++

    def discover(self):
        return self.interface.discover()

    def lock(self):
        return self.interface.lock()

    def release(self):
        return self.interface.release()

    # ++++++++++++++++++++++++++++++++++++++++
    # ItfEmulator methods
    # ++++++++++++++++++++++++++++++++++++++++

    def __getattr__(self, name):
        """
        Relay fall-back attribute getter.
        """
        if hasattr(self.interface, name):
            return getattr(self.interface, name)
        else:
            raise AttributeError(
                "'{:s}' object has no attribute '{:s}'"
                .format(self.__class__.__name__, name)
            )
