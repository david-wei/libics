
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
        0: 0,#drv.DRV_SPAN.AVERAGE_MODE.LIN,
        1: 1#drv.DRV_SPAN.AVERAGE_MODE.EXP
    }

    def __init__(self, cfg):
        super().__init__(cfg)

    def read_powerspectraldensity(self, read_meta=True):
        self._interface.send("SPEC? 0")
        pwr_spectrum = re.split(",", self._interface.recv())[:-1]
        if read_meta:
            self.cfg.bandwidth.val = self._read_bandwidth()
            self.cfg.frequency_start.val = self._read_frequency_start()
            self.cfg.frequency_stop.val = self._read_frequency_stop()
            self.cfg.average_mode.val = self._read_average_mode()
            self.cfg.average_count.val = self._read_average_count()
            self.cfg.voltage_max.val = self._read_voltage_max()
        self._interface.send("UNIT? 0")
        unit = StanfordSR760.UNIT[int(self._interface.recv())]

        psd = arraydata.ArrayData()
        psd.add_dim(
            offset=self.cfg.frequency_start.val,
            scale=((self.cfg.frequency_stop.val - self.cfg.frequency_start.val)
                   / 399),
            name="frequency",
            symbol="f",
            unit="Hz"
        )
        psd.add_dim(
            name="power spectral density",
            symbol="PSD",
            unit=unit
        )
        psd.data = np.array(pwr_spectrum, dtype=float)
        return psd

    def read_spectraldensity(self, read_meta=True):
        pass

    # ++++ Write/read methods +++++++++++

    def _read_bandwidth(self):
        self._interface.send("SPAN?")
        f_range = StanfordSR760.FREQUENCY_SPAN[int(self._interface.recv())]
        return f_range / 399

    def _read_frequency_start(self):
        self._interface.send("STRF?")
        return float(self._interface.recv())

    def _read_frequency_stop(self):
        f_start = self._read_frequency_start()
        self._interface.send("SPAN?")
        f_range = StanfordSR760.FREQUENCY_SPAN[int(self._interface.recv())]
        return f_start + f_range

    def _read_average_mode(self):
        self._interface.send("AVGM?")
        return StanfordSR760.AVERAGING_MODE[int(self._interface.recv())]

    def _write_average_count(self, value):
        if value == 1:
            self._interface.send("AVGO0")
        else:
            self._interface.send("AVGO1")
            self._interface.send("NAVG{:d}".format(int(value)))

    def _read_average_count(self):
        self._interface.send("AVGO?")
        if int(self._interface.recv()) == 1:
            self._interface.send("NAVG?")
            return int(self._interface.recv())
        else:
            return 1

    def _write_voltage_max(self, value):
        self._interface.send("IRNG{:.0f}".format(value))

    def _read_voltage_max(self):
        self._interface.send("IRNG?")
        return float(self._interface.recv())
