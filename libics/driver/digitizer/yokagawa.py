class YokagawaAQ6315(SpAnDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg)

    def read_powerspectraldensity(self, read_meta=True):
        pass

    def read_spectraldensity(self, read_meta=True):
        self._interface.send("DDATA R1-R1001")
        spectrum = [float(x) for x in self._interface.recv().split(", ")][1:]
        if read_meta:
            self.cfg.frequency_start.val = self._read_frequency_start()
            self.cfg.frequency_stop.val = self._read_frequency_stop()

        sp = arraydata.ArrayData()
        sp.data = np.array(spectrum)
        sp.scale.add_dim(
            offset=self.cfg.frequency_start,
            scale=((self.cfg.frequency_stop - self.cfg.frequency_start)
                   / (len(spectrum) - 1)),
            name="spectral density",
            symbol="S",
            unit="rel."
        )
        sp.set_max()
        return sp

    # ++++ Write/read methods +++++++++++

    def _read_frequency_start(self):
        self._interface.send("STAWL?")
        return constants.speed_of_light / float(self._interface.recv())

    def _read_frequency_stop(self):
        self._interface.send("STPWL?")
        return constants.speed_of_light / float(self._interface.recv())
