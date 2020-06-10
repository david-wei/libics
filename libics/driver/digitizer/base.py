

class DRV_SPAN:

    class AVERAGE_MODE:

        LIN = 0
        EXP = 1


class SpAnCfg():

    """
    DrvCfgBase -> SpAnCfg.

    Parameters
    ----------
    bandwidth : float
        Spectral bandwidth in Hertz (Hz).
    frequency_start, frequency_stop : float
        Frequency range (start, stop) in Hertz (Hz).
    average_mode : DRV_SPAN.AVERAGE_MODE
        Averaging mode.
    average_count : int
        Number of averages.
    voltage_max : float
        Voltage input max range in decibel volts (dBV).
    """

    def __init__(
        self,
        bandwidth=1e3,
        frequency_start=0.0, frequency_stop=1e5,
        average_mode=DRV_SPAN.AVERAGE_MODE.LIN, average_count=100,
        voltage_max=-30.0,
        cls_name="SpAnCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.SPAN
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
        self.bandwidth = bandwidth
        self.frequency_start = frequency_start
        self.frequency_stop = frequency_stop
        self.average_mode = average_mode
        self.average_count = average_count
        self.voltage_max = voltage_max

    def get_hl_cfg(self):
        return self




###############################################################################


def get_span_drv(cfg):
    if cfg.model == drv.DRV_MODEL.STANFORD_SR760:
        return StanfordSR760(cfg)
    elif cfg.model == drv.DRV_MODEL.YOKAGAWA_AQ6315:
        return YokagawaAQ6315(cfg)
    else:
        return SpAnDrvBase(cfg)


class SpAnDrvBase():

    def __init__(self, cfg):
        assert(isinstance(cfg, drv.SpAnCfg))
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def read_powerspectraldensity(self, read_meta=True):
        """read_meta flag determines whether to perform a read call for the
        metadata accompanying the spectral data (e.g. frequency)"""
        pass

    @abc.abstractmethod
    def read_spectraldensity(self, read_meta=True):
        """read_meta flag determines whether to perform a read call for the
        metadata accompanying the spectral data (e.g. frequency)"""
        pass

    # ++++ Write/read methods +++++++++++

    def _write_bandwidth(self, value):
        pass

    def _read_bandwidth(self):
        pass

    def _write_frequency_start(self, value):
        pass

    def _read_frequency_start(self):
        pass

    def _write_frequency_stop(self, value):
        pass

    def _read_frequency_stop(self):
        pass

    def _write_average_mode(self, value):
        pass

    def _read_average_mode(self):
        pass

    def _write_average_count(self, value):
        pass

    def _read_average_count(self):
        pass

    def _write_voltage_max(self, value):
        pass

    def _read_voltage_max(self):
        pass


###############################################################################




class OscCfg():

    """
    DrvCfgBase -> OscCfg.

    Parameters
    ----------
    """

    def __init__(
        self,
        cls_name="OscCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.OSC
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)

    def get_hl_cfg(self):
        return self
