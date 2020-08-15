import numpy as np
import re

from libics.env import logging
from libics.core.data.arrays import ArrayData
from libics.driver.digitizer import FftAnalyzer
from libics.driver.digitizer import AVERAGE_MODE, INPUT_COUPLING, FFT_WINDOW
from libics.driver.device import STATUS
from libics.driver.terminal import ItfTerminal


###############################################################################


class SPECTRUM_UNIT:

    VPK = "VPK"
    VRMS = "VRMS"
    DB_VPK = "DB_VPK"
    DB_VRMS = "DB_VRMS"


###############################################################################
# Device
###############################################################################


class StanfordSR760(FftAnalyzer):

    """
    Properties
    ----------
    spectrum_unit : `SPECTRUM_UNIT`
        In-device unit used to measure spectrum.
    device_name : `str`
        Manufacturer device ID.
    """

    # API
    SPECTRUM_UNIT = SPECTRUM_UNIT()

    LOGGER = logging.get_logger(
        "libics.driver.digitizer.stanford.StanfordSR760"
    )

    def __init__(self):
        super().__init__()
        self.properties.set_properties(**self._get_default_properties_dict(
            "spectrum_unit", "device_name"
        ))
        self._is_running = False

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
        self.read_device_name()
        self.p.read_all()

    def close(self):
        self.interface.deregister(id=self.identifier)
        self.interface.close()

    def is_connected(self):
        return self.interface.is_connected()

    # ++++++++++++++++++++++++++++++++++++++++
    # DevSpectrum methods
    # ++++++++++++++++++++++++++++++++++++++++

    def run(self):
        self.interface.send("STRT")
        self._is_running = True

    def stop(self):
        self.interface.send("STCO")
        self._is_running = False

    def is_running(self):
        return self._is_running

    # ++++++++++++++++++++++++++++++++++++++++
    # FftAnalyzer methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_voltage_spectrum(self):
        self.interface.send("MEAS -1 0")
        spectrum = self._read_spectral_data()
        # Set up ArrayData
        ad = ArrayData()
        ad.set_data_quantity(name="voltage spectrum", symbol="U", unit="V")
        ad.add_dim(1)
        ad.set_var_quantity(0, name="frequency", symbol="f", unit="Hz")
        ad.set_dim(
            0, low=self.p.frequency_range[0], high=self.p.frequency_range[1]
        )
        ad.data = spectrum
        return ad

    def read_voltage_spectral_density(self):
        self.interface.send("MEAS -1 1")
        spectrum = self._read_spectral_data()
        # Set up ArrayData
        ad = ArrayData()
        ad.set_data_quantity(
            name="voltage spectral density", symbol="U", unit="V/√Hz"
        )
        ad.add_dim(1)
        ad.set_var_quantity(0, name="frequency", symbol="f", unit="Hz")
        ad.set_dim(
            0, low=self.p.frequency_range[0], high=self.p.frequency_range[1]
        )
        ad.data = spectrum
        return ad

    # ++++++++++++++++++++++++++++++++++++++++
    # StanfordSR760 methods
    # ++++++++++++++++++++++++++++++++++++++++

    def reset(self):
        self.interface.send("*RST")

    # ++++++++++++++++++++++++++++++++++++++++
    # Helper methods
    # ++++++++++++++++++++++++++++++++++++++++

    def _read_spectral_data(self):
        """
        Returns
        -------
        spectrum : `np.ndarray(1, float)`
            Requested raw spectral data in V or V/√Hz.
        """
        spectrum = re.split(",", self.interface.query("SPEC? -1"))[:-1]
        spectrum = np.array(spectrum, dtype=float)
        # Convert unit
        if self.p.spectrum_unit == SPECTRUM_UNIT.VPK:
            pass
        elif self.p.spectrum_unit == SPECTRUM_UNIT.VRMS:
            spectrum *= np.sqrt(2)
        elif self.p.spectrum_unit == SPECTRUM_UNIT.DB_VPK:
            spectrum = 10**(spectrum / 20)
        elif self.p.spectrum_unit == SPECTRUM_UNIT.DB_VRMS:
            spectrum = 10**(spectrum / 20) * np.sqrt(2)
        else:
            raise RuntimeError(
                "invalid unit ({:s})".format(str(self.p.spectrum_unit))
            )
        return spectrum

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    def read_frequency_range(self):
        freq_start = float(self.interface.query("STRF?"))
        MAP = {
            0: 191e-3,   1: 382e-3,     2: 763e-3,     3: 1.5,
            4: 3.1,      5: 6.1,        6: 12.2,       7: 24.4,
            8: 48.75,    9: 97.5,      10: 195.0,     11: 390.0,
            12: 780.0,  13: 1.56e3,    14: 3.125e3,   15: 6.25e3,
            16: 12.5e3, 17: 25e3,      18: 50e3,      19: 100e3
        }
        freq_span = MAP[int(self.interface.query("SPAN?"))]
        value = (freq_start, freq_start + freq_span)
        self.p.frequency_range = value
        return value

    def write_frequency_range(self, value):
        freq_start, freq_span = value[0], value[1] - value[0]
        MAP = {
            191e-3: 0,   382e-3: 1,     763e-3: 2,     1.5: 3,
            3.1: 4,      6.1: 5,        12.2: 6,       24.4: 7,
            48.75: 8,    97.5: 9,       195.0: 10,     390.0: 11,
            780.0: 12,   1.56e3: 13,    3.125e3: 14,   6.25e3: 15,
            12.5e3: 16,  25e3: 17,      50e3: 18,      100e3: 19
        }
        # Find closest suitable frequency range
        dev_spans, dev_codes = np.array([k, v] for k, v in MAP.items()).T
        dev_order = np.argsort(np.abs(dev_spans - freq_span))
        dev_valid = (dev_spans[dev_order] * 1.01 >= freq_span)    # mach. prec.
        idx_suitable = None
        for idx in dev_order:
            if dev_valid[idx]:
                idx_suitable = idx
                break
        if idx_suitable is None:
            self.LOGGER.warning(
                "invalid frequency_range {:s}".format(str(value))
            )
            idx_suitable = dev_order[0]
        span_code = dev_codes[idx_suitable]
        value = (freq_start, freq_start + dev_spans[idx_suitable])
        # Send command
        self.interface.send("STRF {:.5f}".format(freq_start))
        self.interface.send("SPAN {:d}".format(span_code))
        self.p.frequency_range = value

    def read_frequency_bins(self):
        value = 400
        self.p.frequency_bins = value
        return value

    def write_frequency_bins(self, value):
        if value != self.p.frequency_bins:
            self.LOGGER.warning("cannot write frequency_bins")

    def read_average_mode(self):
        value = int(self.interface.query("AVGM?"))
        MAP = {
            0: AVERAGE_MODE.LINEAR,
            1: AVERAGE_MODE.EXPONENTIAL
        }
        value = MAP[value]
        self.p.average_mode = value
        return value

    def write_average_mode(self, value):
        MAP = {
            AVERAGE_MODE.LINEAR: 0,
            AVERAGE_MODE.EXPONENTIAL: 1,
        }
        self.interface.send("AVGM {:d}".format(MAP[value]))
        self.p.average_mode = value

    def read_average_count(self):
        if int(self.interface.query("AVGO?")) == 1:
            value = int(self.interface.query("NAVG?"))
        else:
            value = 1
        self.p.average_count = value
        return value

    def write_average_count(self, value):
        if value == 0 or value == 1:
            self.interface.send("AVGO 0")
        else:
            self.interface.send("AVGO 1")
            self.interface.send("NAVG {:d}".format(int(value)))
        self.p.average_count = value

    def read_input_coupling(self):
        value = int(self.interface.query("ICPL?"))
        MAP = {
            0: INPUT_COUPLING.AC,
            1: INPUT_COUPLING.DC
        }
        value = MAP[value]
        self.p.input_coupling = value
        return value

    def write_input_coupling(self, value):
        MAP = {
            INPUT_COUPLING.AC: 0,
            INPUT_COUPLING.DC: 1
        }
        self.interface.send("ICPL {:d}".format(MAP[value]))
        self.p.input_coupling = value

    def read_voltage_max(self):
        value = float(self.interface.query("IRNG?"))
        # Convert dBV to V
        value = 10**(value / 10)
        self.p.voltage_max = value
        return value

    def write_voltage_max(self, value):
        if value is None:
            self.interface.send("ARNG 1")
        else:
            if int(self.interface.query("ARNG?")) == 1:
                self.interface.send("ARNG 0")
            # Convert V to dBV
            value = int(np.ceil(10 * np.log10(value)))
            self.interface.send("IRNG {:d}".format(value))
        self.p.voltage_max = value

    def read_fft_window(self):
        value = int(self.interface.query("WNDO? -1"))
        MAP = {
            0: FFT_WINDOW.UNIFORM,
            1: FFT_WINDOW.FLATTOP,
            2: FFT_WINDOW.HANNING,
            3: FFT_WINDOW.BLACKMAN_HARRIS
        }
        value = MAP[value]
        self.p.fft_window = value
        return value

    def write_fft_window(self, value):
        MAP = {
            FFT_WINDOW.UNIFORM: 0,
            FFT_WINDOW.FLATTOP: 1,
            FFT_WINDOW.HANNING: 2,
            FFT_WINDOW.BLACKMAN_HARRIS: 3
        }
        self.interface.send("WNDO -1 {:d}".format(MAP[value]))
        self.p.fft_window = value

    def read_spectrum_unit(self):
        value = int(self.interface.query("UNIT? -1"))
        MAP = {
            0: SPECTRUM_UNIT.VPK,
            1: SPECTRUM_UNIT.VRMS,
            2: SPECTRUM_UNIT.DB_VPK,
            3: SPECTRUM_UNIT.DB_VRMS
        }
        value = MAP[value]
        self.p.spectrum_unit = value
        return value

    def write_spectrum_unit(self, value):
        MAP = {
            SPECTRUM_UNIT.VPK: 0,
            SPECTRUM_UNIT.VRMS: 1,
            SPECTRUM_UNIT.DB_VPK: 2,
            SPECTRUM_UNIT.DB_VRMS: 3
        }
        self.interface.send("UNIT -1 {:d}".format(MAP[value]))
        self.p.spectrum_unit = value

    def read_device_name(self):
        value = self.interface.query("*IDN?")
        self.p.device_name = value
        return value

    def write_device_name(self, value):
        if value != self.p.device_name:
            self.LOGGER.warning("cannot write device_name")
