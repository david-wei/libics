class LaserCfg():

    """
    DrvCfgBase -> LaserCfg.

    Parameters
    ----------
    current : float
        Current in Ampere (A).
    temperature : float
        Temperature in degrees (°C).
    """

    def __init__(
        self, current=0.1, temperature=25.0,
        cls_name="LaserCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.LASER
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
        self.current = current
        self.temperature = temperature

    def get_hl_cfg(self):
        return self




###############################################################################


def get_laser_drv(cfg):
    if cfg.model == drv.DRV_MODEL.IPG_YLR:
        return IpgYLR(cfg)


class LaserDrvBase():

    """
    Laser driver API.
    """

    def __init__(self, cfg):
        assert(isinstance(cfg, drv.LaserCfg))
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def write_power(self, power):
        pass

    @abc.abstractmethod
    def read_power(self):
        pass

    @abc.abstractmethod
    def write_current(self, current):
        pass

    @abc.abstractmethod
    def read_current(self):
        pass

    # ++++ Write/read methods +++++++++++

    def _write_current(self, value):
        pass

    def _read_current(self):
        pass

    def _write_temperature(self, value):
        pass

    def _read_temperature(self):
        pass
