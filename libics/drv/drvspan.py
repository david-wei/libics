# System Imports
import abc
import numpy as np
import re

# Package Imports
from libics.data import arraydata
from libics.drv import drv, itf


###############################################################################


def get_span_drv(cfg):
    if cfg.model == drv.DRV_MODEL.STANFORD_SR760:
        return StanfordSR760(cfg)


class SpAnDrvBase(abc.ABC):

    def __init__(self, cfg=None):
        self.cfg = cfg

    @abc.abstractmethod
    def setup(self):
        pass

    @abc.abstractmethod
    def shutdown(self):
        pass

    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def write(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def get_powerspectraldensity(self):
        pass


###############################################################################


class StanfordSR760(SpAnDrvBase):

    MEASUREMENT = {
        0: "spectrum",
        1: "power spectral density",
        2: "time record",
        3: "octave"
    }

    FREQUENCY_SPAN = {
        0: 191e-3,
        1: 382e-3,
        2: 763e-3,
        3: 1.5,
        4: 3.1,
        5: 6.1,
        6: 12.2,
        7: 24.4,
        8: 48.75,
        9: 97.5,
        10: 195.0,
        11: 390.0,
        12: 780.0,
        13: 1.56e3,
        14: 3.125e3,
        15: 6.25e3,
        16: 12.5e3,
        17: 25e3,
        18: 50e3,
        19: 100e3
    }

    UNIT = {
        0: "Vpk/√Hz",
        1: "V/√Hz",
        2: "dBVpk/√Hz",
        3: "dBV/√Hz"
    }

    AVERAGING_TYPE = {
        0: "rms",
        1: "vector",
        2: "peak hold"
    }

    AVERAGING_MODE = {
        0: drv.DRV_SPAN.AVERAGE_MODE.LIN,
        1: drv.DRV_SPAN.AVERAGE_MODE.EXP
    }

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self._interface = None
        self.setup()

    def setup(self, cfg=None):
        if cfg is not None:
            self.cfg = cfg
        self._interface = itf.get_itf(cfg.interface)
        self._interface.setup()

    def shutdown(self):
        self._interface.shutdown()

    def connect(self):
        self._interface.connect()

    def close(self):
        self._interface.close()

    def write(self):
        pass

    def read(self):
        pass

    def get_powerspectraldensity(self):
        # self._interface.send("MEAS 1")    # Measure psd
        # self._interface.send("STRT")      # Start measurement
        # TODO: wait for measurement to finish
        self._interface.send("SPEC? 0")
        pwr_spectrum = re.split(",", self._interface.recv())[:-1]
        self._interface.send("STRF?")
        f_start = float(self._interface.recv())
        self._interface.send("SPAN?")
        f_range = StanfordSR760.FREQUENCY_SPAN[int(self._interface.recv())]
        self._interface.send("UNIT?")
        unit = StanfordSR760.UNIT[int(self._interface.recv())]

        psd = arraydata.ArrayData()
        psd.data = np.array(pwr_spectrum)
        psd.data.scale.add_dim(
            f_start, f_range / (len(pwr_spectrum) - 1),
            name="power spectral density", symbol="PSD", unit=unit
        )
        psd.set_max()
        return psd
