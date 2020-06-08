import abc
import threading


###############################################################################


class TYPES:

    SERIAL = "SERIAL"
    ETHERNET = "ETHERNET"
    USB = "USB"


class STATUS:

    OK = "OK"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    MSG = "MSG"

    ERR_INSTANCE = "ERR_INSTANCE"
    ERR_CONNECTION = "ERR_CONNECTION"
    ERR_INTERFACE = "ERR_INTERFACE"


###############################################################################


class ItfBase(abc.ABC):

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
    def setup(self):
        """
        Instantiates the interface.
        """

    @abc.abstractmethod
    def shutdown(self):
        """
        Destroys the interface.
        """

    @abc.abstractmethod
    def is_setup(self):
        """
        Checks whether interface is instantiated.
        """

    @abc.abstractmethod
    def connect(self):
        """
        Opens the interface.
        """

    @abc.abstractmethod
    def close(self):
        """
        Closes the interface.
        """

    @abc.abstractmethod
    def is_connected(self):
        """
        Checks whether interface is connected.
        """

    @abc.abstractmethod
    def discover(self):
        """
        Discovers devices using the interface.

        Returns
        -------
        devs : `list`
            List of devices available on the interface.
        """

    @abc.abstractmethod
    def status(self):
        """
        Gets the status of the interface.

        Returns
        -------
        status : `dict(STATUS->str)`
            Dictionary containing status information.
            State flags: `OK, ERROR, CRITICAL -> ""`.
            Message: `MSG -> str`.
            Error type: `ERR_INSTANCE, ERR_CONNECTION, ERR_INTERFACE -> ""`.

        Notes
        -----
        Handle the status return by checking the state flags. If the state
        is `CRITICAL`, stop communicating with the interface, delete all
        references and restart the interface from scratch.
        Check the message (value of `MSG` key) for verbose status description).
        For more detailed error handling (to determine which interface parts
        might need to be reset), check the error type flag.
        """

    def recover(self, status=None):
        """
        Recovers the interface after an error.

        Parameters
        ----------
        status : `STATUS`
            Status of interface. If not given, diagnoses `self` using
            the :py:meth:`status` method.
        """
        if status is None:
            status = self.status()
        if STATUS.OK in status:
            return
        is_setup, is_connected = self.is_setup(), self.is_connected()
        if is_connected:
            self.close()
        if is_setup or (
            STATUS.CRITICAL in status
            or STATUS.ERR_INSTANCE in status
            or STATUS.ERR_INTERFACE in status
        ):
            self.shutdown()
            self.setup()
        if is_connected:
            self.connect()

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
