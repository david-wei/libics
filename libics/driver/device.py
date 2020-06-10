import abc


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

    def __init__(self):
        pass

    @abc.abstractmethod
    def setup(self):
        """
        Instantiates the device.
        """
        pass

    @abc.abstractmethod
    def shutdown(self):
        """
        Destroys the device.
        """
        pass

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
        pass

    @abc.abstractmethod
    def close(self):
        """
        Closes connection to the device.
        """
        pass

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


###############################################################################

###############################################################################
###############################################################################
###############################################################################











class DrvBase(abc.ABC):

    """
    Driver base class.

    Provides an API to communicate with the external interface. Communication
    is serialized with a message queue that can be processed. Immediate
    thread-safe communication is enabled by acquiring the `interface_access`
    lock and calling the `read`/`write` methods directly.

    The initialization functions (`setup`, `shutdown`, `connect`, `close`) have
    to be implemented as well as the actual functional methods. Methods using
    the message queue system have to implement I/O functions for each attribute
    and should be named `_write_<attr_name>` and `_read_<attr_name>`. The
    write function takes one value parameter and the read function returns a
    corresponding value parameter. When implementing direct access methods,
    guard the method with lock access.

    Parameters
    ----------
    cfg : DrvBaseCfg
        Driver configuration object.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg
        self._interface = None
        self.interface_access = threading.Lock()

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
        self.close()
        self.shutdown()

    def setup(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg
        self._interface = drv.itf.get_itf(self.cfg.interface)
        self._interface.setup()

    def shutdown(self):
        self._interface.shutdown()

    def connect(self):
        self._interface.connect()

    def close(self):
        self._interface.close()

    def write(self, msg):
        func = getattr(self, "_write_" + msg.name)
        self.interface_access.acquire()
        try:
            if msg.value is None:
                func()
            else:
                func(msg.value)
        except cfg.err.RUNTM_DRV as e:
            print(e)
        finally:
            self.interface_access.release()

    def read(self, msg):
        func = getattr(self, "_read_" + msg.name)
        self.interface_access.acquire()
        ret = None
        try:
            ret = func()
        except cfg.err.RUNTM_DRV as e:
            print(e)
        finally:
            self.interface_access.release()
        if ret is not None:
            msg.callback(ret)

    def process(self):
        """
        Processes the message queue in the configuration object.
        """
        msg = self.cfg._pop_msg()
        while (msg is not None):
            if (msg.msg_type == cfg.CFG_MSG_TYPE.WRITE or
                    msg.msg_type == cfg.CFG_MSG_TYPE.VALIDATE):
                self.write(msg)
            if (msg.msg_type == cfg.CFG_MSG_TYPE.READ or
                    msg.msg_type == cfg.CFG_MSG_TYPE.VALIDATE):
                self.read(msg)
            msg = self.cfg._pop_msg()

    def write_all(self):
        """
        Writes all configuration items from interface.
        Automatically processes the message queue.
        """
        self.cfg.write_all()
        self.process()

    def read_all(self):
        """
        Reads all configuration items from interface.
        Automatically processes the message queue.
        """
        self.cfg.read_all()
        self.process()

    def get_drv(self, cfg=None):
        if cfg is None:
            cfg = self.cfg
        return drv.get_drv(cfg)


###############################################################################


class DRV_DRIVER:

    CAM = 0         # Camera
    PIEZO = 10      # Piezo controller
    PICO = 11       # Pico motor controller
    SPAN = 20       # Spectrum analyzer
    OSC = 30        # Oscilloscope
    DSP = 40        # Display
    LASER = 50      # Laser


class DRV_MODEL:

    # Cam
    ALLIEDVISION_MANTA_G145B_NIR = 101
    VRMAGIC_VRMCX = 201

    # Piezo
    THORLABS_MDT69XA = 1101
    THORLABS_MDT693A = THORLABS_MDT69XA
    THORLABS_MDT694A = THORLABS_MDT69XA

    # Pico
    NEWPORT_8742 = 1201

    # SpAn
    STANFORD_SR760 = 2101
    YOKAGAWA_AQ6315 = 2201
    AGILENT_N9320X = 2301

    # Osc
    TEKTRONIX_TDS100X = 3101

    # Dsp
    TEXASINSTRUMENTS_DLP7000 = 4101

    # Laser
    IPG_YLR = 5101


class DrvCfgBase():

    """
    DrvCfgBase.

    Parameters
    ----------
    driver : DRV_DRIVER
        Driver type.
    interface : drv.itf.itf.ProtocolCfgBase
        Connection interface configuration.
    identifier : str
        Unique identifier of device.
    model : DRV_MODEL
        Device model.
    """

    def __init__(
        self,
        driver=DRV_DRIVER.CAM, interface=None, identifier="", model="",
        cls_name="DrvCfgBase", **kwargs
    ):
        super().__init__(cls_name=cls_name)
        self.driver = driver
        self.identifier = identifier
        self.model = model
        self._kwargs = kwargs
        if isinstance(interface, dict):
            self.interface = (
                drv.itf.itf.ProtocolCfgBase(**interface).get_hl_cfg()
            )
        else:
            self.interface = interface

    def get_hl_cfg(self):
        MAP = {
            DRV_DRIVER.CAM: CamCfg,
            DRV_DRIVER.PIEZO: PiezoCfg,
            DRV_DRIVER.PICO: PicoCfg,
            DRV_DRIVER.SPAN: SpAnCfg,
            DRV_DRIVER.OSC: OscCfg,
            DRV_DRIVER.DSP: DspCfg,
            DRV_DRIVER.LASER: LaserCfg,
        }
        obj = MAP[self.driver.val](ll_obj=self, **self._kwargs)
        return obj.get_hl_cfg()

    def write_all(self):
        """
        Set all configuration items by writing all values to interface.
        """
        for key, val in self.__dict__.items():
            if isinstance(val, cfg.CfgItem):
                val.write()

    def read_all(self):
        """
        Update all configuration items by reading all values from interface.
        """
        for key, val in self.__dict__.items():
            if isinstance(val, cfg.CfgItem):
                val.read()
