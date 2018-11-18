from libics import cfg
from libics.data.types import Quantity, ValQuantity
from libics.util import misc, InheritMap


###############################################################################


@InheritMap(map_key=("libics", "SetupCfgBase"))
class SetupCfgBase(cfg.CfgBase):

    """
    Container class for storing unstructured flat data.

    Attributes are stored as object attributes in the object dictionary. If
    not explicitly specified, the attribute values must be of
    `data.types.ValQuantity` type.
    """

    def __init__(self, **attrs):
        super().__init__(cls_name="SetupCfgBase")
        for key, val in attrs.items():
            if isinstance(val, ValQuantity):
                pass
            elif isinstance(val, Quantity):
                attrs[key] = ValQuantity(
                    name=val.name, symbol=val.symbol, unit=val.unit
                )
            elif isinstance(val, dict):
                if "name" not in val:
                    val["name"] = key
                attrs[key] = misc.assume_construct_obj(val, ValQuantity)
            else:
                attrs[key] = ValQuantity(name=key, val=val)
        self.__dict__.update(attrs)

    def get_hl_cfg(self):
        return self


###############################################################################


@InheritMap(map_key=("libics", "ModulationCfg"))
class ModulationCfg(SetupCfgBase):

    """
    Container class for signal modulation.

    Parameters
    ----------
    device_name : ValQuantity
        val: modulator model name.
    method : ValQuantity
        val: modulation method, e.g. AM, FM, etc.
    magnitude : ValQuantity
        val: modulation magnitude, e.g. amplitude in AM
             or bandwidth in FM.
        unit: unit as appropriate.
    rate : ValQuantity
        val: modulation rate, e.g. frequency of AM/FM.
        unit: Hertz (Hz).
    """

    def __init__(
        self,
        method=None, magnitude=None, rate=None,
        **attrs
    ):
        super().__init__(
            method=method, magnitude=magnitude, rate=rate, **attrs
        )


@InheritMap(map_key=("libics", "SignalCfg"))
class SignalCfg(SetupCfgBase):

    """
    Container class for (electric) signal sources.

    Parameters
    ----------
    device_name : ValQuantity
        val: signal generator model name.
    mixer_name : ValQuantity
        val: mixer model name.
    amplifier_name : ValQuantity
        val: amplifier model name.
    carrier_magnitude : ValQuantity
        name: carrier voltage or power.
        val: carrier magnitude (dc/rms).
        unit: Volt (V) or Watt (W).
    carrier_frequency : ValQuantity
        val: carrier frequency.
        unit: Hertz (Hz).
    amplitude_mod : ModulationCfg
        Amplitude modulation of signal.
    frequency_mod : ModulationCfg
        Frequency modulation of signal.
    """

    def __init__(
        self,
        device_name=None, mixer_name=None, amplifier_name=None,
        carrier_magnitude=None, carrier_frequency=None,
        amplitude_mod=None, frequency_mod=None,
        **attrs
    ):
        super().__init__(
            device_name=device_name, mixer_name=mixer_name,
            amplifier_name=amplifier_name, carrier_magnitude=carrier_magnitude,
            carrier_frequency=carrier_frequency,
            **attrs
        )
        self.amplitude_mod = misc.assume_construct_obj(
            amplitude_mod, ModulationCfg
        )
        self.frequency_mod = misc.assume_construct_obj(
            frequency_mod, ModulationCfg
        )


@InheritMap(map_key=("libics", "PhotodiodeCfg"))
class PhotodiodeCfg(SetupCfgBase):

    """
    Container class for photodiode properties.

    Parameters
    ----------
    device_name : ValQuantity
        val: photodiode model name.
    gain : ValQuantity
        val: photodiode gain.
        unit: Decibel (dB).
    voltage_dc : ValQuantity
        val: photodiode DC signal voltage.
        unit: Volt (V).
    """

    def __init__(
        self,
        device_name=None, gain=None, voltage_dc=None, **attrs
    ):
        super().__init__(
            device_name=device_name, gain=gain, voltage_dc=voltage_dc, **attrs
        )


@InheritMap(map_key=("libics", "AcoustoOpticModulatorCfg"))
class AcoustoOpticModulatorCfg(SetupCfgBase):

    """
    Container class for acousto optic modulator properties.

    Parameters
    ----------
    device_name : ValQuantity
        val: AOM model name.
    rf_driver : SignalCfg
        Amplified radio-frequency driving signal source.
    """

    def __init__(
        self, device_name=None, rf_driver=None, **attrs
    ):
        super().__init__(
            device_name=device_name, **attrs
        )
        self.rf_driver = misc.assume_construct_obj(rf_driver, SignalCfg)


@InheritMap(map_key=("libics", "PIDControllerCfg"))
class PIDControllerCfg(SetupCfgBase):

    """
    Container class for PID controller properties.

    Parameters
    ----------
    device_name : ValQuantity
        PID controller model name.
    process : SetupCfgBase or ValQuantity
        PID process variable device (feedback).
    setpoint : ModulationCfg or ValQuantity
        PID setpoint variable device (feedback target value).
    control : SetupCfgBase or ValQuantity
        PID control variable device (control target).
    """

    # TODO: add PID parameters

    def __init__(
        self,
        device_name=None, process=None, setpoint=None, control=None,
        **attrs
    ):
        try:
            self.process = misc.assume_construct_obj(
                process, SetupCfgBase, raise_exception=ValueError
            )
        except ValueError:
            attrs["process"] = process
        try:
            self.setpoint = misc.assume_construct_obj(
                setpoint, ModulationCfg, raise_exception=ValueError
            )
        except ValueError:
            attrs["setpoint"] = setpoint
        try:
            self.control = misc.assume_construct_obj(
                control, SetupCfgBase, raise_exception=ValueError
            )
        except ValueError:
            attrs["control"] = control
        super().__init__(device_name=device_name, **attrs)
