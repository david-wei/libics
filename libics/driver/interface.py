import abc







###############################################################################


class ITF_PROTOCOL:

    """
    Interface protocol types.

    Attributes
    ----------
    TEXT:
        Text-based protocol, e.g. used for RS-232.
    BINARY:
        Binary protocol, e.g. used in IPC for Vimba.
    """

    TEXT = 0
    BINARY = 1


@InheritMap(map_key=("libics", "ProtocolCfgBase"))
class ProtocolCfgBase(cfg.CfgBase):

    """
    Interface configuration base class.

    Parameters
    ----------
    protocol : ITF_PROTOCOL (int)
        Interface protocol type.
    interface : ITF_TXT or ITF_BIN
        Interface class.

    Notes
    -----
    * If None parameters are passed, the respective attributes are interpreted
      as unspecified.
    * Derived classes should implement the get_hl_cfg method returning the
      next higher level configuration class.
    * Derived classes should pass initialization arguments to base class
      constructors and check for base class objects.
    """

    def __init__(self, protocol=ITF_PROTOCOL.TEXT, interface=None,
                 cls_name="ProtocolCfgBase", **kwargs):
        super().__init__(cls_name=cls_name)
        self.protocol = protocol
        self.interface = interface
        self._kwargs = kwargs

    def get_hl_cfg(self):
        """
        Gets the higher level configuration object.

        Returns
        -------
        obj : class derived from ProtocolCfgBase
            Highest level object.

        Raises
        ------
        KeyError:
            If highest level object could not be constructed.
        """
        MAP = {
            ITF_PROTOCOL.TEXT: TxtCfgBase,
            ITF_PROTOCOL.BINARY: BinCfgBase
        }
        obj = MAP[self.protocol](ll_obj=self, **self._kwargs)
        return obj.get_hl_cfg()


###############################################################################


class ITF_TXT:

    """
    Text-based interface protocols.

    Attributes
    ----------
    SERIAL:
        Serial interface.
    ETHERNET:
        Ethernet interface.
    """

    SERIAL = 11
    ETHERNET = 12
    USB = 13


@InheritMap(map_key=("libics", "TxtCfgBase"))
class TxtCfgBase(ProtocolCfgBase):

    """
    ProtocolCfgBase -> TxtCfgBase.

    Parameters
    ----------
    address : str
        Address of interface, e.g. IP.
    buffer_size : int
        Number of buffer bytes.
    send_timeout : float
        Send: timeout in seconds.
    send_termchar : str
        Send: command termination characters.
    recv_timeout : float
        Receive: timeout in seconds.
    recv_termchar : str
        Receive: command termination characters.
    """

    def __init__(self, address=None, buffer_size=None,
                 send_timeout=None, send_termchar=None,
                 recv_timeout=None, recv_termchar=None,
                 cls_name="TxtCfgBase", ll_obj=None, **kwargs):
        if "protocol" not in kwargs.keys():
            kwargs["protocol"] = ITF_PROTOCOL.TEXT
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.address = address
        self.buffer_size = buffer_size
        self.send_timeout = send_timeout
        self.send_termchar = send_termchar
        self.recv_timeout = recv_timeout
        self.recv_termchar = recv_termchar

    def get_hl_cfg(self):
        MAP = {
            ITF_TXT.SERIAL: TxtSerialCfg,
            ITF_TXT.ETHERNET: TxtEthernetCfg,
            ITF_TXT.USB: TxtUsbCfg,
        }
        obj = MAP[self.interface](ll_obj=self, **self._kwargs)
        return obj.get_hl_cfg()


# +++++++++++++++++++++++++++++


class ITF_BIN:

    """
    Binary interface protocols.

    Attributes
    ----------
    VIMBA:
        AlliedVision Vimba.
    VRMAGIC:
        VRmagic USB.
    VIALUX:
        Vialux ALP4.x.
    """

    VIMBA = 101
    VRMAGIC = 102
    VIALUX = 111


@InheritMap(map_key=("libics", "BinCfgBase"))
class BinCfgBase(ProtocolCfgBase):

    """
    ProtocolCfgBase -> BinCfgBase.

    Parameters
    ----------
    device : str
        String identifier for device (cf. address).
        None typically searches for interfaces and automatically
        chooses one (depending on implementation).
    """

    def __init__(self, device=None,
                 cls_name="BinCfgBase", ll_obj=None, **kwargs):
        if "protocol" not in kwargs.keys():
            kwargs["protocol"] = ITF_PROTOCOL.BINARY
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.device = device

    def get_hl_cfg(self):
        MAP = {
            ITF_BIN.VIMBA: BinVimbaCfg,
            ITF_BIN.VRMAGIC: BinVRmagicCfg,
            ITF_BIN.VIALUX: BinVialuxCfg
        }
        obj = MAP[self.interface](ll_obj=self, **self._kwargs)
        return obj.get_hl_cfg()







###############################################################################


class ItfBase(abc.ABC):

    DEVICES = set()

    def __init__(self):
        pass

    @abc.abstractmethod
    def configure(self, **cfg):
        """
        Configures the interface.
        """
        pass

    @abc.abstractmethod
    def setup(self):
        """
        Instantiates the interface.
        """
        pass

    @abc.abstractmethod
    def shutdown(self):
        """
        Destroys the interface.
        """
        pass

    @abc.abstractmethod
    def connect(self):
        """
        Opens the interface.
        """
        pass

    @abc.abstractmethod
    def close(self):
        """
        Closes the interface.
        """
        pass

    @abc.abstractmethod
    def discover(self):
        """
        Discovers devices using the interface.
        """
        pass

    @abc.abstractmethod
    def status(self):
        """
        Gets the status of the interface.
        """
        pass

    @abc.abstractmethod
    def recover(self):
        """
        Recovers the interface after an error.
        """
        pass

    @abc.abstractmethod
    def lock(self):
        """
        Locks access to the interface.
        """
        pass

    @abc.abstractmethod
    def release(self):
        """
        Releases access lock to the interface.
        """
        pass


###############################################################################


class ItfTerminal(ItfBase):

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def send(self, msg):
        """
        Sends a message.
        """
        pass

    @abc.abstractmethod
    def recv(self):
        """
        Receives a message.
        """
        pass

    def query(self, msg):
        """
        Sends and receives a message.
        """
        self.send(msg)
        return self.recv()

    def validate(self, msg, val):
        """
        Sends a message and checks for validation response.
        """
        self.send(msg)
        return self.recv() == val

    @abc.abstractmethod
    def _trim(self, msg):
        """
        Pre-processes a received message.
        """
        pass

    @abc.abstractmethod
    def flush_out(self):
        """
        Flushes the output buffer.
        """
        pass

    @abc.abstractmethod
    def flush_in(self):
        """
        Flushes the input buffer.
        """
        pass

    @abc.abstractmethod
    def ping(self):
        """
        Pings a device.
        """
        pass
