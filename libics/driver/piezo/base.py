


class DRV_PIEZO:

    class FEEDBACK_MODE:

        OPEN_LOOP = 0
        CLOSED_LOOP = 1


class PiezoCfg():

    """
    DrvCfgBase -> PiezoCfg.

    Parameters
    ----------
    limit_min, limit_max : float
        Voltage limit minimum and maximum in volts (V).
    displacement : float
        Displacement in meters per volt (m/V).
    channel : int or str
        Voltage channel.
    feedback_mode : DRV_PIEZO.FEEDBACK_MODE
        Feedback operation mode.
    """

    def __init__(
        self,
        limit_min=0.0, limit_max=75.0,
        displacement=2.67e-7,
        channel=None,
        feedback_mode=DRV_PIEZO.FEEDBACK_MODE.OPEN_LOOP,
        cls_name="PiezoCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.PIEZO
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.displacement = displacement
        self.channel = channel
        self.feedback_mode = feedback_mode

    def get_hl_cfg(self):
        return self


class DRV_PICO:

    class FEEDBACK_MODE:

        OPEN_LOOP = 0
        CLOSED_LOOP = 1


class PicoCfg():

    """
    DrvCfgBase -> PicoCfg.

    Parameters
    ----------
    channel : int or tuple(int)
        If `int`, indicates pico motor channel.
        If `tuple(int)`, represents slave device tree
        (slave level 0, slave level 1, ..., motor channel).
    acceleration : int
        Picomotor acceleration in steps per square seconds (st/s²).
    velocity : int
        Picomotor velocity in steps per second (st/s).
    feedback_mode : DRV_PIEZO.FEEDBACK_MODE
        Feedback operation mode.
    """
    
    def __init__(
        self,
        acceleration=100000, velocity=1750,
        channel=1,
        feedback_mode=DRV_PICO.FEEDBACK_MODE.OPEN_LOOP,
        cls_name="PicoCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.PIEZO
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
        self.channel = channel
        self.acceleration = acceleration
        self.velocity = velocity
        self.feedback_mode = feedback_mode

    def get_hl_cfg(self):
        return self



###############################################################################


def get_piezo_drv(cfg):
    if cfg.model == drv.DRV_MODEL.THORLABS_MDT69XA:
        return ThorlabsMDT69XA(cfg)


class PiezoDrvBase():

    """
    Piezo driver API.
    """

    def __init__(self, cfg):
        assert(isinstance(cfg, drv.PiezoCfg))
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def write_voltage(self, voltage):
        pass

    @abc.abstractmethod
    def read_voltage(self):
        pass

    # ++++ Write/read methods +++++++++++

    def _write_limit_min(self, value):
        pass

    def _read_limit_min(self):
        pass

    def _write_limit_max(self, value):
        pass

    def _read_limit_max(self):
        pass

    def _write_displacement(self, value):
        pass

    def _read_displacement(self):
        pass

    def _write_channel(self, value):
        pass

    def _read_channel(self):
        pass

    def _write_feedback_mode(self, value):
        pass

    def _read_feedback_mode(self):
        pass






def get_pico_drv(cfg):
    if cfg.model == drv.DRV_MODEL.NEWPORT_8742:
        return Newport8742(cfg)


class PicoDrvBase():

    """
    Picomotor driver API.
    """

    def __init__(self, cfg):
        assert(isinstance(cfg, drv.PicoCfg))
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def abort_motion(self):
        pass

    @abc.abstractmethod
    def zero_position(self):
        pass

    @abc.abstractmethod
    def move_relative(self, steps):
        pass

    # ++++ Write/read methods +++++++++++

    def _write_feedback_mode(self, value):
        pass

    def _read_feedback_mode(self):
        pass
