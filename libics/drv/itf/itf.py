# Package Imports
from libics import cfg


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

    def __init__(self, protocol=ITF_PROTOCOL.TEXT, interface=None, **kwargs):
        super().__init__()
        self.protocol = protocol
        self.interface = interface
        self.kwargs = kwargs

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
        obj = MAP[self.protocol](ll_obj=self, **self.kwargs)
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
                 ll_obj=None, **kwargs):
        super().__init__(**kwargs)
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
            ITF_TXT.ETHERNET: TxtEthernetCfg
        }
        obj = MAP[self.interface](ll_obj=self, **self.kwargs)
        return obj.get_hl_cfg()


# +++++++++++++++++++++++++++++


class ITF_BIN:

    """
    Binary interface protocols.

    Attributes
    ----------
    VIMBA:
        AlliedVision Vimba.
    """

    VIMBA = 101


class BinCfgBase(ProtocolCfgBase):

    """
    ProtocolCfgBase -> BinCfgBase.

    Parameters
    ----------
    device : str
        String identifier for device (cf. address).
    """

    def __init__(self, device=None, ll_obj=None, **kwargs):
        super().__init__(**kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.device = device

    def get_hl_cfg(self):
        MAP = {
            ITF_BIN.VIMBA: BinVimbaCfg,
        }
        obj = MAP[self.interface](ll_obj=self, **self.kwargs)
        return obj.get_hl_cfg()


###############################################################################


class TxtSerialCfg(TxtCfgBase):

    """
    ProtocolCfgBase -> TxtCfgBase -> TxtSerialCfg.

    Parameters
    ----------
    baudrate : int
        Baud rate in bits per second.
    bytesize : 5, 6, 7, 8
        Number of data bits.
    parity : "none", "even", "odd", "mark", "space"
        Parity checking.
    stopbits : 1, 1.5, 2
        Number of stop bits.
    """

    def __init__(self, baudrate=115200, bytesize=8, parity="none",
                 stopbits=1,
                 ll_obj=None, **kwargs):
        super().__init__(**kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits

    def get_hl_cfg(self):
        return self


class TxtEthernetCfg(TxtCfgBase):

    """
    ProtocolCfgBase -> TxtCfgBase -> TxtEthernetCfg.

    Parameters
    ----------
    port : int
        Ethernet port number.
    """

    def __init__(self, port=None, blocking=False, ll_obj=None, **kwargs):
        super().__init__(**kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)
        self.port = port
        self.blocking = blocking

    def get_hl_cfg(self):
        return self


class BinVimbaCfg(BinCfgBase):

    def __init__(self, ll_obj=None, **kwargs):
        super().__init__(**kwargs)
        if ll_obj is not None:
            self.__dict__.update(ll_obj.__dict__)

    def get_hl_cfg(self):
        return self
