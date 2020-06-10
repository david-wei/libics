import abc

from libics.core.env import logging
from libics.core.util.func import StoppableThread


###############################################################################


class STATUS:

    """
    Device status container class.

    Parameters
    ----------
    state : `str`
        State of device (`OK, ERROR, CRITICAL`).
    err_type : `str`
        Error type (`ERR_NONE, ERR_INSTANCE, ERR_CONNECTION, ERR_DEVICE`).
    msg : `str`
        Error message.
    """

    OK = "OK"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    ERR_NONE = "ERR_NONE"
    ERR_INSTANCE = "ERR_INSTANCE"
    ERR_CONNECTION = "ERR_CONNECTION"
    ERR_DEVICE = "ERR_DEVICE"

    def __init__(self, state="OK", err_type="ERR_NONE", msg=None):
        self.state = None
        self.err_type = None
        self.msg = None
        self.set_state(state)
        self.set_err_type(err_type)
        self.set_message(msg)

    def is_ok(self):
        return self.state == self.OK

    def is_critical(self):
        return self.state == self.CRITICAL

    def has_err_type(self, *err_types):
        return self.err_type in err_types

    def get_state(self):
        return self.state

    def set_state(self, state):
        if state in (self.OK, self.ERROR, self.CRITICAL):
            self.state = state
            if state == self.OK:
                self.set_err_type(self.ERR_NONE)
        else:
            raise ValueError("invalid state ({:s})".format(str(state)))

    def get_err_type(self):
        return self.err_type

    def set_err_type(self, err_type):
        if self.is_ok():
            if err_type != self.ERR_NONE:
                raise ValueError("invalid err_type ({:s})".format(err_type))
        else:
            if err_type not in (
                self.ERR_INSTANCE, self.ERR_CONNECTION, self.ERR_DEVICE
            ):
                raise ValueError("invalid err_type ({:s})".format(err_type))
        self.err_type = err_type

    def has_message(self):
        return self.msg is not None

    def get_message(self):
        return "" if self.msg is None else self.msg

    def set_message(self, msg):
        self.msg = msg

    def __str__(self):
        s = self.get_state()
        if not self.is_ok():
            s += " ({:s})".format(self.get_err_type())
        if self.has_message():
            s += ": {:s}".format(self.get_message())
        return s

    def __repr__(self):
        return "<device.STATUS> {:s}".format(str(self))


###############################################################################


class DevBase(abc.ABC):

    LOGGER = logging.get_logger("libics.driver.device.DevBase")

    def __init__(self):
        self.identifier = None
        self.model = None
        self.interface = None
        self.properties = DevProperties()
        self.properties.set_device(self)

    @property
    def i(self):
        return self.interface

    @property
    def p(self):
        return self.properties

    @abc.abstractmethod
    def setup(self):
        """
        Instantiates the device.
        """

    @abc.abstractmethod
    def shutdown(self):
        """
        Destroys the device.
        """

    @abc.abstractmethod
    def is_set_up(self):
        """
        Checks whether device is instantiated.
        """

    @abc.abstractmethod
    def connect(self):
        """
        Connects to the device.
        """

    @abc.abstractmethod
    def close(self):
        """
        Closes connection to the device.
        """

    @abc.abstractmethod
    def is_connected(self):
        """
        Checks whether device is connected.
        """

    @abc.abstractmethod
    def status(self):
        """
        Gets the status of the device.

        Returns
        -------
        status : `STATUS`
            Device status information.

        Notes
        -----
        Handle the status return by checking the state flags. If the state
        is `CRITICAL`, stop communicating with the interface, delete all
        references and restart the interface from scratch.
        Check the message for verbose status description.
        For more detailed error handling (to determine which interface parts
        might need to be reset), check the error type.
        """

    def recover(self, status=None):
        """
        Recovers the device after an error.

        Parameters
        ----------
        status : `driver.device.STATUS`
            Status of interface. If not given, diagnoses `self` using
            the :py:meth:`status` method.
        """
        if status is None:
            status = self.status()
        if status.is_ok():
            return
        is_set_up, is_connected = self.is_set_up(), self.is_connected()
        if is_connected:
            self.close()
        if is_set_up and (
            status.is_critical()
            or status.has_err_type(STATUS.ERR_INSTANCE, STATUS.ERR_DEVICE)
        ):
            self.shutdown()
            self.setup()
        if is_connected:
            self.connect()

    def __enter__(self):
        try:
            self.setup()
        except Exception:
            try:
                self.shutdown()
            except Exception:
                pass
            raise
        try:
            self.connect()
        except Exception:
            try:
                self.close()
            except Exception:
                pass
            raise
        return self

    def __exit__(self, *args):
        if self.is_connected():
            self.close()
        if self.is_set_up():
            self.shutdown()


###############################################################################


class DevProperties(object):

    """
    Dynamical in-device properties.

    Provides a unified API to access device functions.
    """

    READ = 0
    WRITE = 1

    def __init__(self):
        self.dev = None
        self.props = dict()
        self.__thread = None

    @property
    def LOGGER(self):
        return self.dev.LOGGER

    def set_device(self, dev):
        self.dev = dev

    def set_properties(self, **props):
        """
        Updates properties.

        Parameters
        ----------
        **props : `str->(callable, callable)`
            Property name -> (read function, write function).
            Read function call signature: `read()->object`.
            Write function call signature: `write(object)`.
            If `None` is given instead of a `callable`, no external device
            communication is used and the locally saved value is used
            (i.e. instead of read/write get/set is used).
        """
        for prop, funcs in props.items():
            if funcs is None:
                self.props[prop] = (
                    lambda: getattr(self, prop),
                    lambda val: setattr(self, prop, val)
                )
            elif len(funcs) == 2:
                self.props[prop] = (
                    funcs[0] if callable(funcs[0])
                    else lambda: getattr(self, prop),
                    funcs[1] if callable(funcs[0])
                    else lambda val: setattr(self, prop, val)
                )
            else:
                raise ValueError("invalid property function ({:s}->{:s})"
                                 .format(prop, str(funcs)))

    def rmv_properties(self, *props):
        """
        Removes properties.

        Parameters
        ----------
        *props : `str`
            Property name. If none are given, removes all.
        """
        if len(props) == 0:
            self.LOGGER.info("removing all properties")
            for prop in self.props:
                delattr(self, prop)
            self.props = dict()
        else:
            for prop in props:
                delattr(self, prop)
                del self.props[prop]

    def get(self, *props):
        """
        Gets locally saved property.

        Parameters
        ----------
        *props : `str`
            Property name.

        Returns
        -------
        vals : `object` or `dict(str->object)`
            Property value(s).
        """
        if len(props) == 1:
            if props[0] in self.props:
                return getattr(self, props[0])
            else:
                self.LOGGER.error("invalid property ({:s})".format(props[0]))
                return None
        else:
            return [self.get(prop) for prop in props]

    def set(self, **props):
        """
        Locally sets property. The value is NOT written to the device!

        Parameters
        ----------
        **props : `str->object`
            Property name -> value.
        """
        for prop, val in props.items():
            if prop in self.props:
                setattr(self, prop, val)
            else:
                self.LOGGER.error("invalid property ({:s})".format(prop))

    def read(self, *props):
        """
        Reads property from device.

        Parameters
        ----------
        *props : `str`
            Property name.

        Returns
        -------
        vals : `object` or `dict(str->object)`
            Property value(s).
        """
        if len(props) == 1:
            if props[0] in self.props:
                val = self.props[props[0]][self.READ]()
                setattr(self, props[0], val)
                return val
            else:
                self.LOGGER.error("invalid property ({:s})".format(props[0]))
                return None
        else:
            return [self.read(prop) for prop in props]

    def write(self, **props):
        """
        Writes property to device.

        Parameters
        ----------
        **props : `str->object`
            Property name -> value.
        """
        for prop, val in props.items():
            if prop in self.props:
                if getattr(self, prop) != val:
                    self.props[prop][self.WRITE](val)
                    setattr(self, prop, val)
            else:
                self.LOGGER.error("invalid property ({:s})".format(prop))

    def apply(self, **props):
        """
        Reads property from device.
        If different, writes the given value to device.

        Parameters
        ----------
        **props : `str->object`
            Property name -> value.
        """
        for prop, val in props.items():
            if prop in self.props:
                ret_val = self.props[prop][self.READ]()
                if ret_val != val:
                    self.props[prop][self.WRITE](val)
                    setattr(self, prop, val)
            else:
                self.LOGGER.error("invalid property ({:s})".format(prop))

    def runp(self, func, *args, **kwargs):
        """
        Wrapper for a parallel thread running a worker function.

        Can be used to asynchronously perform slow device I/O.

        Parameters
        ----------
        func : `callable`
            Worker function to be started in thread.
        *args, **kwargs
            (Keyword) arguments passed on to the worker function.

        Raises
        ------
        RuntimeError
            If a thread is already running.
        """
        if self.__thread is not None:
            if self.__thread.is_alive():
                raise RuntimeError("thread active")
            else:
                self.__thread = None
        self.__thread = StoppableThread(target=func, args=args, kwargs=kwargs)
        self.__thread.start()

    def stopp(self):
        """
        Stops a previously run parallel thread.
        """
        if self.__thread is not None:
            self.__thread.stop()
        self.__thread = None
