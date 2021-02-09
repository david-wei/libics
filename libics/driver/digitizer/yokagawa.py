import numpy as np

from libics.env import constants
from libics.core.data.arrays import ArrayData
from libics.driver.digitizer import SpectrumAnalyzer
from libics.driver.digitizer import AVERAGE_MODE
from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal


###############################################################################
# Device
###############################################################################


class YokagawaAQ6315(SpectrumAnalyzer):

    def __init__(self):
        super().__init__()

    # ++++++++++++++++++++++++++++++++++++++++
    # Device methods
    # ++++++++++++++++++++++++++++++++++++++++

    def setup(self):
        """
        Raises
        ------
        RuntimeError
            If interface is not set.
        """
        if not isinstance(self.interface, ItfTerminal):
            err_msg = "invalid interface"
            self.last_status = STATUS(
                state=STATUS.ERROR, err_type=STATUS.ERR_CONNECTION, msg=err_msg
            )
            raise RuntimeError(err_msg)
        self.interface.setup()

    def shutdown(self):
        if self.is_connected():
            self.close()
        if self.is_set_up():
            self.interface.shutdown()

    def is_set_up(self):
        return self.interface.is_set_up()

    def connect(self):
        if not self.is_set_up():
            raise RuntimeError("device not set up")
        self.interface.connect(self.identifier)
        self.interface.register(self.identifier, self)
        self.p.read_all()

    def close(self):
        self.interface.deregister(id=self.identifier)
        self.interface.close()

    def is_connected(self):
        return self.interface.is_connected()

    # ++++++++++++++++++++++++++++++++++++++++
    # DevSpectrum methods
    # ++++++++++++++++++++++++++++++++++++++++

    def run(self, single=False):
        """
        Parameters
        ----------
        single : `bool`
            Flag whether to measure only once.
            Otherwise measures continuously.
        """
        # Use trace A
        self.interface.send("WRTA")
        self.interface.send("ACTV 0")
        if single:
            self.interface.send("SGL")
        else:
            self.interface.send("RPT")

    def stop(self):
        self.interface.send("STP")

    def is_running(self):
        value = int(self.interface.query("SWEEP?"))
        return value != 0

    # ++++++++++++++++++++++++++++++++++++++++
    # SpectrumAnalyzer methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_power_spectrum(self):
        """
        Returns
        -------
        ad : `data.arrays.ArrayData`
            Power spectrum.
            Dimensions: [frequency (Hz)]->power spectrum (rel.).
            Normalization by discrete sum.
        """
        self._write_data_readout_preparation()
        wl = self._read_wavelength_data().flip()
        freq = constants.c / wl
        spectrum = self._read_spectral_data().flip()
        spectrum /= np.sum(spectrum)
        # Set up ArrayData
        ad = ArrayData()
        ad.set_data_quantity(
            name="power spectrum", symbol="P", unit="\\mathrm{{rel.}}"
        )
        ad.add_dim(1)
        ad.set_var_quantity(
            0, name="frequency", symbol="f", unit="Hz"
        )
        ad.set_dim(0, points=freq)
        ad.data = spectrum
        return ad

    def read_power_spectral_density(self):
        """
        Returns
        -------
        ad : `data.arrays.ArrayData`
            Power spectrum.
            Dimensions: [frequency (Hz)]->PSD (rel./Hz).
            Normalization by integral.
        """
        self._write_data_readout_preparation()
        wl = self._read_wavelength_data().flip()
        freq = constants.c / wl
        df = np.mean(freq[1:] - freq[:-1])
        psd = self._read_spectral_data().flip()
        psd /= np.sum(psd)
        psd /= df
        # Set up ArrayData
        ad = ArrayData()
        ad.set_data_quantity(
            name="power spectral density", symbol="P",
            unit="\\mathrm{{rel.}} / Hz"
        )
        ad.add_dim(1)
        ad.set_var_quantity(
            0, name="frequency", symbol="f", unit="Hz"
        )
        ad.set_dim(0, points=freq)
        ad.data = psd
        return ad

    # ++++++++++++++++++++++++++++++++++++++++
    # Helper methods
    # ++++++++++++++++++++++++++++++++++++++++

    def _write_data_readout_preparation(self):
        """
        Prepares the device for data readout.
        """
        self.interface.send("SD 0")
        self.interface.send("BD 0")
        self.interface.send("HD 1")

    def _read_wavelength_data(self):
        """
        Returns
        -------
        wl : `np.ndarray(1, float)`
            Wavelengths in meter (m).
        """
        ret = self.interface.query("WDATA R1-R1001").split(" ")
        header = ret[0]
        assert(header.upper() == "NM")
        data = np.array([float(item) for item in ret[-1].split(",")[1:]])
        return data * 1e-9

    def _read_spectral_data(self):
        """
        Returns
        -------
        spectrum : `np.ndarray(1, float)`
            Requested raw spectral data in arbitrary units.
        """
        ret = self.interface.query("LDATA R1-R1001").split(" ")
        header = ret[0]
        data = np.array([float(item) for item in ret[-1].split(",")[1:]])
        # Convert units
        if header[:2].upper() == "DB":
            data = 10**(data / 10) * 1e-3
        return data

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_frequency_range(self):
        freq_stop = constants.c / float(self.interface.query("STAWL?")) * 1e9
        freq_start = constants.c / float(self.interface.query("STPWL?")) * 1e9
        value = (freq_start, freq_stop)
        self.p.frequency_range = value
        return value

    def write_frequency_range(self, value):
        freq_start, freq_stop = value
        wl_start, wl_stop = constants.c / freq_stop, constants.c / freq_start
        self.interface.send("STAWL {:.2f}".format(wl_start * 1e9))
        self.interface.send("STPWL {:.2f}".format(wl_stop * 1e9))
        self.p.frequency_range = value

    def read_frequency_bins(self):
        wl_span = (
            constants.c / self.p.frequency_range[0]
            - constants.c / self.p.frequency_range[1]
        )
        wl_resln = int(self.interface.query("RESLN?")) * 1e-9
        value = int(round(wl_span / wl_resln)) + 1
        self.p.frequency_bins = value
        return value

    def write_frequency_bins(self, value):
        wl_span = (
            constants.c / self.p.frequency_range[0]
            - constants.c / self.p.frequency_range[1]
        )
        wl_resln = wl_span / (value - 1)
        self.interface.send("RESLN {:.2f}".format(
            round(20 * wl_resln * 1e9) / 20
        ))
        self.p.frequency_bins = value

    def read_average_mode(self):
        value = AVERAGE_MODE.EXPONENTIAL
        self.p.average_mode = value
        return value

    def write_average_mode(self, value):
        if value != self.p.average_mode:
            self.LOGGER.warning("cannot write average_mode")

    def read_average_count(self):
        value = int(self.interface.query("AVG?"))
        self.p.average_count = value
        return value

    def write_average_count(self, value):
        self.interface.send("AVG {:d}".format(int(value)))
        self.p.average_count = value

    def read_power_max(self):
        value = self.interface.query("REFL?").split(" ")
        # Convert units
        if len(value) == 1:     # dBm
            value = 10**(float(value[0]) / 10) * 1e-3
        else:
            MAP = {"PW": 1e-12, "NW": 1e-9, "UW": 1e-6, "MW": 1e-3}
            value = float(value[1]) * MAP[value[0]]
        self.p.power_max = value
        return value

    def write_power_max(self, value):
        if value is None:
            self.interface.send("ATREF 1")
        else:
            if int(self.interface.query("ATREF?")) == 1:
                self.interface.send("ATREF 0")
            # Convert units
            if value >= 1e-3:
                self.interface.send("REFLM {:.2f}".format(value * 1e3))
            elif value >= 1e-6:
                self.interface.send("REFLU {:.2f}".format(value * 1e6))
            elif value >= 1e-9:
                self.interface.send("REFLN {:.2f}".format(value * 1e9))
            elif value >= 1e-12:
                self.interface.send("REFLP {:.2f}".format(value * 1e12))
        self.p.power_max = value
