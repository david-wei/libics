# System Imports
import numpy as np
import re

# Package Imports
from libics.cfg import err as ERR
from libics.util.types import FlaggedType

# Subpackage Imports
from libics.drv.itf import gpib


###############################################################################


class SpecAnalyzerCfg(object):

    class Device:

        DEVICE_TYPE = ["sr760"]

        def __init__(self, device_type="sr760", device_id=0,
                     ipv4="130.183.96.12", port=1234):
            self.device_type = FlaggedType(
                device_type, cond=SpecAnalyzerCfg.Device.DEVICE_TYPE
            )
            self.device_id = FlaggedType(device_id)
            self.ipv4 = FlaggedType(ipv4)
            self.port = FlaggedType(port)

    def __init__(self, device_type="sr760", device_id=0,
                 ipv4="130.183.96.12", port=1234):
        self.device = SpecAnalyzerCfg.Device(
            device_type=device_type, device_id=device_id,
            ipv4=ipv4, port=port
        )

    def set_all_flags(self, flag):
        """
        Sets the flags of all attributes to the given boolean value.
        """
        for cat_key, category in self.__dict__.items():
            if cat_key != "camera":
                for _, item in category.__dict__.items():
                    item.flag = flag

    def set_config(self, specanalyzer_cfg, flags=None):
        """
        Sets the configuration parameters.

        If an attribute of the passed `specanalyzer_cfg` is `None`, this value
        is not set.

        Parameters
        ----------
        specanalyzer_cfg : SpecAnalyzerCfg
            The spectrum analyzer configuration to be set.
        flags : None or bool
            `None`: Sets differential update flags.
            `True`: Sets all update flags.
            `False`: Sets no update flags.
        """
        diff_flag = (flags is None)
        for cat_key, cat_val in self.__dict__.items():
            if cat_key != "device":
                for item_key, item_val in cat_val.__dict__.items():
                    cam_cfg_item_val = (specanalyzer_cfg.__dict__[cat_key]
                                        .__dict__[item_key])
                    if cam_cfg_item_val is not None:
                        if item_val is None:
                            item_val = cam_cfg_item_val.copy()
                        elif cam_cfg_item_val != item_val:
                            item_val.assign(cam_cfg_item_val,
                                            diff_flag=diff_flag)
                        else:
                            item_val.flag = False
        if type(flags) == bool:
            self.set_all_flags(flags)


###############################################################################


class SpecAnalyzer(object):

    """
    Function call distribution wrapper.

    Depending on which spectrum analyzer driver is opened, different setup and
    control functions are called to obtain the same behaviour despite different
    (hardware) interfaces.

    Parameters
    ----------
    specanalyzer_cfg : SpecAnalyzerCfg
        Spectrum analyzer configuration container determining driver settings.

    Raises
    ------
    cfg.err.RUNTM_DRV_SPA
        If spectrum analyzer runtime error occurs.
    """

    def __init__(self, specanalyzer_cfg=SpecAnalyzerCfg()):
        self._specanalyzer_cfg = specanalyzer_cfg
        self._specanalyzer_itf = None

    def setup_specanalyzer(self):
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            self._specanalyzer_itf = _setup_specanalyzer_sr760(
                self._specanalyzer_cfg
            )
        else:
            raise ERR.RUNTM_DRV_SPA(ERR.RUNTM_DRV_SPA.str())

    def shutdown_specanalyzer(self):
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            pass
        else:
            raise ERR.RUNTM_DRV_SPA(ERR.RUNTM_DRV_SPA.str())

    # ++++ Spectrum analyzer connection ++++++++++++++++

    def get_specanalyzer(self):
        return self._specanalyzer_itf

    def open_specanalyzer(self):
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            _open_specanalyzer_sr760(self._specanalyzer_itf)

    def close_specanalyzer(self):
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            self._specanalyzer_itf.close()

    # ++++ Spectrum analyzer configuration +++++++++++++

    def set_specanalyzer_cfg(self, specanalyzer_cfg):
        self._specanalyzer_cfg.set_config(specanalyzer_cfg)

    def get_specanalyzer_cfg(self):
        return self._specanalyzer_cfg

    def read_specanalyzer_cfg(self, overwrite_cfg=False):
        specanalyzer_cfg = None
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            specanalyzer_cfg = self._specanalyzer_cfg
        if overwrite_cfg:
            self.set_specanalyzer_cfg(specanalyzer_cfg)
        return specanalyzer_cfg

    def write_specanalyzer_cfg(self):
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            pass

    # ++++ Spectrum analyzer control +++++++++++++++++++

    def read_spectrum(self):
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            return _read_spectrum_sr760(self._specanalyzer_itf)

    def read_unit(self):
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            return _read_unit_sr760(self._specanalyzer_itf)

    def read_averaging(self):
        if self._specanalyzer_cfg.device.device_type.val == "sr760":
            return _read_averaging_sr760(self._specanalyzer_itf)


###############################################################################
# Initialization
###############################################################################

# ++++++++++ SR 760 ++++++++++++++++++++++++++++++


def _setup_specanalyzer_sr760(specanalyzer_cfg):
    ipv4 = specanalyzer_cfg.device.ipv4.val
    port = specanalyzer_cfg.device.port.val
    connection = gpib.ConnectionGpibEthernet(ipv4, port)
    specanalyzer_itf = gpib.GPIB(connection=connection)
    return specanalyzer_itf


def _open_specanalyzer_sr760(specanalyzer_itf):
    specanalyzer_itf.connect()
    specanalyzer_itf.send("++savecfg 0")
    specanalyzer_itf.send("++mode 1")
    specanalyzer_itf.send("++eoi 1")
    specanalyzer_itf.send("++auto 0")
    specanalyzer_itf.send("++eos 3")
    specanalyzer_itf.send("++addr 5")
    specanalyzer_itf.send("++read_tmo_ms 1000")


###############################################################################
# Configuration
###############################################################################


###############################################################################
# Control
###############################################################################

# ++++++++++ SR 760 ++++++++++++++++++++++++++++++


def _read_spectrum_sr760(specanalyzer_itf):
    # Read spectrum
    specanalyzer_itf.send("SPEC? 0")
    spectrum = specanalyzer_itf.receive()
    spectrum = re.split(r",", spectrum)
    spectrum = np.array(spectrum[0:len(spectrum) - 1])
    # Read frequencies
    specanalyzer_itf.send("STRF?")
    strf = float(specanalyzer_itf.receive())
    specanalyzer_itf.send("SPAN?")
    span = UtilSr760.FREQUENCY_SPANS[int(specanalyzer_itf.receive())]
    frequencies = np.linspace(strf, strf + span, len(spectrum))
    # Convert data types
    try:
        spectrum = np.array([frequencies, spectrum], dtype="float64")
    except(ValueError):
        raise ERR.RUNTM_DRV_SPA("Cannot interpret spectrum")
    return spectrum


def _read_unit_sr760(specanalyzer_itf):
    specanalyzer_itf.send("UNIT?")
    unit = UtilSr760.DISPLAY_UNITS[int(specanalyzer_itf.receive())]
    return unit


def _read_averaging_sr760(specanalyzer_itf):
    specanalyzer_itf.send("AVGO?")
    is_averaging = int(specanalyzer_itf.receive()) == 1
    specanalyzer_itf.send("NAVG?")
    n_averages = int(specanalyzer_itf.receive())
    specanalyzer_itf.send("AVGT?")
    avg_type = UtilSr760.AVERAGING_TYPES[int(specanalyzer_itf.receive())]
    specanalyzer_itf.send("AVGM?")
    avg_mode = UtilSr760.AVERAGING_MODES[int(specanalyzer_itf.receive())]
    return (is_averaging, n_averages, avg_type, avg_mode)


###############################################################################
# Util
###############################################################################


class UtilSr760:

    FREQUENCY_SPANS = {
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
        10: 195,
        11: 390,
        12: 780,
        13: 1.56e3,
        14: 3.125e3,
        15: 6.25e3,
        16: 12.5e3,
        17: 25e3,
        18: 50e3,
        19: 100e3
    }

    DISPLAY_UNITS = {
        0: "Vpk/sqrt(Hz)",
        1: "Vrms/sqrt(Hz)",
        2: "dBV/sqrt(Hz)",
        3: "dBVrms/sqrt(Hz)"
    }

    AVERAGING_TYPES = {
        0: 'rms',
        1: 'vector',
        2: 'peak hold'
    }

    AVERAGING_MODES = {
        0: 'lin',
        1: 'exp'
    }


###############################################################################


if __name__ == "__main__":

    # Test imports
    import matplotlib.pyplot as plt
    import json
    import os

    # Test configuration
    device_type = "sr760"
    ipv4 = "130.183.96.12"
    port = 1234
    file_dir = os.path.join(os.environ["USERPROFILE"], "Desktop",
                            "20181002_SpecAnalyzer")
    file_name = "specanalyzer_test"
    voltage_dc = 1.0

    # Setup test object
    specanalyzer_cfg = SpecAnalyzerCfg(
        device_type=device_type, ipv4=ipv4, port=port
    )
    specanalyzer = SpecAnalyzer(specanalyzer_cfg=specanalyzer_cfg)
    specanalyzer.setup_specanalyzer()
    specanalyzer.open_specanalyzer()

    # Test functions
    def normalize_spectrum(spectrum, unit, voltage_dc):
        if unit == "Vpk/sqrt(Hz)":
            spectrum[1] = spectrum[1] / voltage_dc / np.sqrt(2)
            unit = "c/sqrt(Hz)"
        elif unit == "Vrms/sqrt(Hz)":
            spectrum[1] = spectrum[1] / voltage_dc
            unit = "c/sqrt(Hz)"
        elif unit == "dBVpk/sqrt(Hz)":
            spectrum[1] = (spectrum[1] - 10 * np.log10(2)
                           - 20 * np.log10(voltage_dc))
            unit = "dBc/Hz"
        elif unit == "dBVrms/sqrt(Hz)":
            spectrum[1] = spectrum[1] - 20 * np.log10(voltage_dc)
            unit = "dBc/Hz"
        return spectrum, unit

    def save_spectrum(file_path, spectrum, unit, averaging):
        d = dict()
        d["spectrum"] = spectrum.tolist()
        d["unit"] = unit
        d["averaging"] = averaging
        with open(os.path.join(file_path), "w") as f:
            json.dump(d, f)

    def load_spectrum(file_path):
        d = dict()
        unit, spectrum, averaging = None, None, None
        with open(file_path, "r") as f:
            d = json.load(f)
        for it in d.items():
            if it[0] == "unit":
                unit = it[1]
            elif it[0] == "spectrum":
                spectrum = np.array(it[1])
            elif it[0] == "averaging":
                averaging = it[1]
        return spectrum, unit, averaging

    def plot_spectrum(spectrum, unit="dBc/Hz", averaging=[]):
        plt.plot(spectrum[0], spectrum[1], label="{:s}".format(str(averaging)))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Spectrum ({:s})".format(unit))
        plt.grid(which="both")

    # Load spectrum and metadata
    spectrum = specanalyzer.read_spectrum()
    unit = specanalyzer.read_unit()
    averaging = specanalyzer.read_averaging()

    # Save, load and plot spectrum
    plot_spectrum(spectrum, unit, averaging)
    _spectrum, _unit = normalize_spectrum(spectrum, unit)
    file_path = os.path.join(file_dir, file_name + ".json")
    save_spectrum(file_path, spectrum, unit, averaging)
    _spectrum, _unit, _averaging = load_spectrum(file_path)
    plot_spectrum(_spectrum, _unit, _averaging)
