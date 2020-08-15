import abc

from libics.driver.device import DevBase


###############################################################################


class AVERAGE_MODE:

    LINEAR = "LINEAR"
    EXPONENTIAL = "EXPONENTIAL"


class INPUT_COUPLING:

    AC = "AC"
    DC = "DC"


class FFT_WINDOW:

    UNIFORM = "UNIFORM"
    FLATTOP = "FLATTOP"
    HANNING = "HANNING"
    BLACKMAN_HARRIS = "BLACKMAN_HARRIS"


###############################################################################


class DevSpectrum(DevBase):

    """
    Properties
    ----------
    frequency_range : `(float, float)`
        Frequency range (start, stop) in Hertz (Hz).
    frequency_bins : `int`
        Number of frequency bins.
    average_mode : `AVERAGE_MODE`
        Averaging mode.
    average_count : `int`
        Number of averages.
    """

    def __init__(self):
        super().__init__()
        self.properties.set_properties(**self._get_default_properties_dict(
            "bandwidth", "frequency_range", "average_mode", "average_count"
        ))

    # ++++++++++++++++++++++++++++++++++++++++
    # DevSpectrum methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def run(self):
        """
        Start measurement.
        """

    @abc.abstractmethod
    def stop(self):
        """
        Stop measurement.
        """

    @abc.abstractmethod
    def is_running(self):
        """
        Checks whether measurement is on-going.
        """

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_frequency_range(self):
        pass

    @abc.abstractmethod
    def write_frequency_range(self, value):
        pass

    @abc.abstractmethod
    def read_frequency_bins(self):
        pass

    @abc.abstractmethod
    def write_frequency_bins(self, value):
        pass

    @abc.abstractmethod
    def read_average_mode(self):
        pass

    @abc.abstractmethod
    def write_average_mode(self, value):
        pass

    @abc.abstractmethod
    def read_average_count(self):
        pass

    @abc.abstractmethod
    def write_average_count(self, value):
        pass


class SpectrumAnalyzer(DevSpectrum):

    """
    Properties
    ----------
    power_max : `float`
        Maximum input power in Watt (W).
    """

    def __init__(self):
        super().__init__()
        self.properties.set_properties(**self._get_default_properties_dict(
            "power_max"
        ))

    # ++++++++++++++++++++++++++++++++++++++++
    # SpectrumAnalyzer methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_power_spectrum(self):
        """
        Reads the power spectrum per frequency bin.

        Returns
        -------
        ad : `data.arrays.ArrayData`
            Power spectrum.
            Dimensions: [frequency (Hz)]->power spectrum (W).
        """

    @abc.abstractmethod
    def read_power_spectral_density(self):
        """
        Reads the frequency-normalized power spectral density.

        Returns
        -------
        ad : `data.arrays.ArrayData`
            Power spectral density.
            Dimensions: [frequency (Hz)]->PSD (W/Hz).
        """

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_power_max(self):
        pass

    @abc.abstractmethod
    def write_power_max(self, value):
        pass


class FftAnalyzer(DevBase):

    """
    Properties
    ----------
    input_coupling : `INPUT_COUPLING`
        Input port coupling (AC/DC).
    voltage_max : `float` or `None`
        Maximum input voltage in Watt (V).
        If `None`, sets value automatically.
    fft_window : `FFT_WINDOW`
        FFT windowing function.
    """

    def __init__(self):
        super().__init__()
        self.properties.set_properties(**self._get_default_properties_dict(
            "voltage_max", "fft_window"
        ))

    # ++++++++++++++++++++++++++++++++++++++++
    # FftAnalyzer methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_voltage_spectrum(self):
        """
        Reads the voltage spectrum per frequency bin.

        Returns
        -------
        ad : `data.arrays.ArrayData`
            Voltage spectrum.
            Dimensions: [frequency (Hz)]->voltage spectrum (V).
        """

    @abc.abstractmethod
    def read_voltage_spectral_density(self):
        """
        Reads the frequency-normalized voltage spectral density.

        Returns
        -------
        ad : `data.arrays.ArrayData`
            Voltage spectral density.
            Dimensions: [frequency (Hz)]->voltage PSD (V/√Hz).
        """

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_input_coupling(self):
        pass

    @abc.abstractmethod
    def write_input_coupling(self, value):
        pass

    @abc.abstractmethod
    def read_voltage_max(self):
        pass

    @abc.abstractmethod
    def write_voltage_max(self, value):
        pass

    @abc.abstractmethod
    def read_fft_window(self):
        pass

    @abc.abstractmethod
    def write_fft_window(self, value):
        pass


###############################################################################


class Oscilloscope(DevBase):

    def __init__(self):
        super().__init__()
        raise NotImplementedError
