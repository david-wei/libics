import abc
import threading

from libics.driver.device import DevBase


###############################################################################


class ItfBase(DevBase):

    """
    Interface base class.

    This class provides an API for interface devices, including virtual
    drivers (from external libraries) or physical devices which act as
    intermediaries (as a means to communicate with another device).
    A unique interface can be used by multiple device, which should
    :py:meth:`register` with the interface. To prevent racing conditions,
    devices should acquire a :py:meth:`lock`.
    """

    DEVICES = dict()

    def __init__(self):
        if self.__class__.__name__ not in self.DEVICES:
            self.DEVICES[self.__class__.__name__] = set()
        self._lock = threading.Lock()

    def configure(self, **cfg):
        """
        Configures the interface.

        Parameters
        ----------
        **cfg
            Keyword arguments setting interface properties.
        """
        for k, v in cfg.items():
            setattr(self, k, v)

    @abc.abstractmethod
    def discover(self):
        """
        Discovers devices using the interface.

        Returns
        -------
        devs : `list`
            List of devices available on the interface.
        """

    def lock(self):
        """
        Locks access to the interface.
        """
        self._lock.acquire()

    def release(self):
        """
        Releases access lock to the interface.
        """
        self._lock.release()

    def register(self, dev):
        """
        Registers a device using the interface.
        """
        self.DEVICES[self.__class__.__name__].add(dev)

    def deregister(self, dev):
        """
        De-registers a device from the interface.
        """
        self.DEVICES[self.__class__.__name__].remove(dev)

    def devices(self):
        """
        Gets all registered devices using the interface.
        """
        return self.DEVICES[self.__class__.__name__]
