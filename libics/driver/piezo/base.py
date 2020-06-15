import abc

from libics.driver.device import DevBase


###############################################################################


class PiezoActuator(DevBase):

    """
    Configuration
    -------------
    channel : `int`
        Voltage channel.
    deformation_coeff : `float`
        Displacement-voltage ratio in meters per volt (m/V).

    Properties
    ----------
    voltage : `float`
        Voltage in volts (V).
    voltage_min, voltage_max : `float`
        Voltage limit minimum and maximum in volts (V).
    """

    def __init__(self):
        super().__init__()
        self.channel = None
        self.deformation_coeff = None
        self.properties.set_properties(self._get_default_properties_dict(
            "voltage", "voltage_min", "voltage_max"
        ))

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_voltage(self):
        pass

    @abc.abstractmethod
    def write_voltage(self, value):
        pass

    @abc.abstractmethod
    def read_voltage_min(self):
        pass

    @abc.abstractmethod
    def write_voltage_min(self, value):
        pass

    @abc.abstractmethod
    def read_voltage_max(self):
        pass

    @abc.abstractmethod
    def write_voltage_max(self, value):
        pass


###############################################################################


class Picomotor(DevBase):

    """
    Configuration
    -------------
    channel : `int`
        Motor axis.
    step_size : `float`
        Displacement per step in meters (m).

    Parameters
    ----------
    position : `int`
        Picomotor position in steps w.r.t. home (st).
    acceleration : `float`
        Picomotor acceleration in steps per square seconds (st/s²).
    velocity : `float`
        Picomotor velocity in steps per second (st/s).
    """

    def __init__(self):
        super().__init__()
        self.channel = None
        self.step_size = None
        self.properties.set_properties(self._get_default_properties_dict(
            "position", "acceleration", "velocity"
        ))

    # ++++++++++++++++++++++++++++++++++++++++
    # Picomotor methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def abort_motion(self):
        pass

    @abc.abstractmethod
    def home_position(self):
        pass

    @abc.abstractmethod
    def move_relative(self, steps):
        pass

    # ++++++++++++++++++++++++++++++++++++++++
    # Properties methods
    # ++++++++++++++++++++++++++++++++++++++++

    @abc.abstractmethod
    def read_position(self):
        pass

    @abc.abstractmethod
    def write_position(self, value):
        pass

    @abc.abstractmethod
    def read_acceleration(self):
        pass

    @abc.abstractmethod
    def write_acceleration(self, value):
        pass

    @abc.abstractmethod
    def read_velocity(self):
        pass

    @abc.abstractmethod
    def write_velocity(self, value):
        pass
