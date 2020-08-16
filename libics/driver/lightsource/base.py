import abc
import time

from libics.driver.device import DevBase


###############################################################################


class LightSource(DevBase):

    """
    Attributes
    ----------

    **Properties**

    current : `float`
        Light source current in Ampere (A).
    power : `float`
        Optical power in Watt (W).
    temperature : `float`
        Light source temperature in Celsius (°C).
    emission_time : `float`
        Emission ("ON") time in seconds (s).
    """

    def __init__(self):
        super().__init__()
        self.properties.set_properties(**self._get_default_properties_dict(
            "current", "power", "temperature", "emission_time"
        ))

    # ++++++++++++++++++++++++++++++++++++++++
    # LightSource methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def run(self):
        """
        Starts light emission.
        """

    @abc.abstractmethod
    def stop(self):
        """
        Stops light emission.
        """

    @abc.abstractmethod
    def is_running(self):
        """
        Checks if emission is active.
        """

    def ramp(self, prop, target_value, steps=16, duration=1.0):
        """
        Linearly ramps a property in discrete steps.

        Parameters
        ----------
        prop : `str`
            Property name to be ramped.
        target_value : `float`
            Ramp target property value.
        steps : `int`
            Number of discrete steps in ramp.
        duration : `float`
            Total ramp duration in seconds (s).
        """
        step_time = duration / steps
        _value = self.p.read(prop)
        step_value = (target_value - _value) / steps
        _time = time.time()
        for i in range(steps):
            _value += step_value
            _time += step_time
            self.p.write(**{prop: _value})
            _duration = _time - time.time()
            if _duration > 0:
                time.sleep(_duration)

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_current(self):
        pass

    @abc.abstractmethod
    def write_current(self, value):
        pass

    @abc.abstractmethod
    def read_power(self):
        pass

    @abc.abstractmethod
    def write_power(self, value):
        pass

    @abc.abstractmethod
    def read_temperature(self):
        pass

    @abc.abstractmethod
    def write_temperature(self, value):
        pass

    @abc.abstractmethod
    def read_emission_time(self):
        pass

    @abc.abstractmethod
    def write_emission_time(self, value):
        pass
