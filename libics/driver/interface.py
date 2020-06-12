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
            self.DEVICES[self.__class__.__name__] = dict()
        self._lock = threading.Lock()
        self.interface = self

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def connect(self, id):
        """
        Connects to the device.

        Parameters
        ----------
        id : `str`
            Interface-unique device ID to connect to.
        """

    @abc.abstractmethod
    def close(self):
        """
        Closes connection to the device.

        Parameters
        ----------
        id : `str`
            Interface-unique device ID to disconnect from.
        """

    # ++++++++++++++++++++++++++++++++++++++++
    # Interface methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def discover(self):
        """
        Discovers devices using the interface.

        Returns
        -------
        devs : `list(str)`
            List of device IDs available on the interface.
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

    def register(self, id, dev):
        """
        Registers a device using the interface.

        Parameters
        ----------
        id : `str`
            Device ID to be registered.
        dev : `object`
            Associated device object to be registered.
        """
        self.DEVICES[self.__class__.__name__][id] = dev

    def deregister(self, id=None, dev=None):
        """
        De-registers a device from the interface either by ID or object.

        Parameters
        ----------
        id : `str`
            Device ID to be deregistered. Takes precedence over `dev`.
        dev : `object`
            Associated device object to be deregistered.
        """
        if id is None:
            for k, v in self.DEVICES[self.__class__.__name__].items():
                if dev == v:
                    id = k
                    break
        del self.DEVICES[self.__class__.__name__][dev]

    def devices(self):
        """
        Gets all registered devices using the interface.
        """
        return self.DEVICES[self.__class__.__name__]
