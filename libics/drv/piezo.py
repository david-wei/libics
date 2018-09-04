# System Imports
import copy

# Package Imports
from libics.cfg import err as ERR
from libics.util.types import FlaggedType

# Subpackage Imports
from libics.drv.itf import mdt693


###############################################################################


class PiezoCfg(object):

    class Device:

        DEVICE_TYPE = ["mdt693"]

        def __init__(self, device_type="mdt693", device_id=0,
                     port=None, timeout_send=1.0, timeout_recv=1.0,
                     hw_proc_delay=0.05):
            self.device_type = FlaggedType(
                device_type, cond=PiezoCfg.Device.DEVICE_TYPE
            )
            self.device_id = FlaggedType(device_id)
            self.port = FlaggedType(port)
            self.timeout_send = FlaggedType(timeout_send)
            self.timeout_recv = FlaggedType(timeout_recv)
            self.hw_proc_delay = FlaggedType(hw_proc_delay)

    class Voltage:

        def __init__(self, voltage_min=0.0, voltage_max=75.0,
                     range_per_volt=2.5e-7):
            self.voltage_min = FlaggedType(voltage_min)
            self.voltage_max = FlaggedType(voltage_max)
            self.range_per_volt = FlaggedType(range_per_volt)

    def __init__(self, device_type="mdt693", device_id=0, port=None):
        self.device = PiezoCfg.Device(
            device_type=device_type, device_id=device_id, port=port
        )
        self.voltage = PiezoCfg.Voltage()

    def set_all_flags(self, flag):
        """
        Sets the flags of all attributes to the given boolean value.
        """
        for cat_key, category in self.__dict__.items():
            if cat_key != "camera":
                for _, item in category.__dict__.items():
                    item.flag = flag

    def set_config(self, piezo_cfg, flags=None):
        """
        Sets the configuration parameters.

        If an attribute of the passed `piezo_cfg` is `None`, this value is
        not set.

        Parameters
        ----------
        piezo_cfg : PiezoCfg
            The camera configuration to be set.
        flags : None or bool
            `None`: Sets differential update flags.
            `True`: Sets all update flags.
            `False`: Sets no update flags.
        """
        diff_flag = (flags is None)
        for cat_key, cat_val in self.__dict__.items():
            if cat_key != "device":
                for item_key, item_val in cat_val.__dict__.items():
                    cam_cfg_item_val = (piezo_cfg.__dict__[cat_key]
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


class Piezo(object):

    """
    Function call distribution wrapper.

    Depending on which piezo driver is opened, different setup and control
    functions are called to obtain the same behaviour despite different
    (hardware) interfaces.

    Parameters
    ----------
    piezo_cfg : PiezoCfg
        Piezo configuration container determining driver settings.

    Raises
    ------
    cfg.err.RUNTM_DRV_PIZ
        If piezo runtime error occurs.
    """

    def __init__(self, piezo_cfg=PiezoCfg()):
        self._piezo_cfg = piezo_cfg
        self._piezo_itf = None

    def setup_piezo(self):
        if self._piezo_cfg.device.device_type.val == "mdt693":
            try:
                self._piezo_itf = _setup_piezo_mdt693(self._piezo_cfg)
            except mdt693.serial.SerialException as e:
                raise ERR.RUNTM_DRV_PIZ(ERR.RUNTM_DRV_PIZ.str(str(e)))
        else:
            raise ERR.RUNTM_DRV_PIZ(ERR.RUNTM_DRV_PIZ.str())

    def shutdown_piezo(self):
        if self._piezo_cfg.device.device_type.val == "mdt693":
            pass
        else:
            raise ERR.RUNTM_DRV_PIZ(ERR.RUNTM_DRV_PIZ.str())

    # ++++ Piezo connection ++++++++++++++++++++++++++++

    def get_piezo(self):
        return self._piezo_itf

    def open_piezo(self):
        if self._piezo_cfg.device.device_type.val == "mdt693":
            try:
                self._piezo_itf.open_serial()
            except mdt693.serial.SerialException as e:
                raise ERR.RUNTM_DRV_PIZ(ERR.RUNTM_DRV_PIZ.str(str(e)))

    def close_piezo(self):
        if self._piezo_cfg.device.device_type.val == "mdt693":
            self._piezo_itf.close_serial()

    # ++++ Camera configuration ++++++++++++++++++++++++

    def set_piezo_cfg(self, piezo_cfg):
        self._piezo_cfg.set_config(piezo_cfg)

    def get_piezo_cfg(self):
        return self._piezo_cfg

    def read_piezo_cfg(self, overwrite_cfg=False):
        piezo_cfg = None
        if self._piezo_cfg.device.device_type.val == "mdt693":
            piezo_cfg = _read_piezo_cfg_mdt693(
                self._piezo_itf, self._piezo_cfg
            )
        if overwrite_cfg:
            self.set_piezo_cfg(piezo_cfg)
        return piezo_cfg

    def write_piezo_cfg(self):
        if self._piezo_cfg.device.device_type.val == "mdt693":
            _write_piezo_cfg_mdt693(self._piezo_itf, self._piezo_cfg)

    # ++++ Piezo control +++++++++++++++++++++++++++++++

    def set_voltage(self, voltage):
        ERR.assertion(ERR.RUNTM_DRV_PIZ,
                      voltage is not None,
                      description="invalid voltage")
        if self._piezo_cfg.device.device_type.val == "mdt693":
            _set_voltage_mdt693(self._piezo_itf, self._piezo_cfg, voltage)

    def get_voltage(self):
        voltage = None
        if self._piezo_cfg.device.device_type.val == "mdt693":
            voltage = _get_voltage_mdt693(self._piezo_itf, self._piezo_cfg)
        return voltage


###############################################################################
# Initialization
###############################################################################

# ++++++++++ MDT 693 +++++++++++++++++++++++++++++


def _setup_piezo_mdt693(piezo_cfg):
    port = piezo_cfg.device.port.val
    # If unspecified, automatically choose first serial port
    if port is None:
        port = mdt693._list_serial_ports()[0]
    piezo_itf = mdt693.MDT693(
        port=port, read_timeout=piezo_cfg.device.timeout_recv.val,
        write_timeout=piezo_cfg.device.timeout_send.val,
        hw_proc_delay=piezo_cfg.device.hw_proc_delay.val
    )
    return piezo_itf


###############################################################################
# Configuration
###############################################################################

# ++++++++++ MDT 693 +++++++++++++++++++++++++++++


def _read_piezo_cfg_mdt693(piezo_itf, piezo_cfg):
    piezo_cfg = copy.deepcopy(piezo_cfg)

    voltage_range = piezo_itf.get_voltage_range(
        channel=piezo_cfg.device.device_id.val
    )
    if piezo_cfg.device.device_id is None:
        voltage_range = [
            min([v_ch[0] for v_ch in voltage_range]),
            max([v_ch[1] for v_ch in voltage_range])
        ]
    piezo_cfg.voltage.voltage_min.val = voltage_range[0]
    piezo_cfg.voltage.voltage_max.val = voltage_range[1]

    return piezo_cfg


def _write_piezo_cfg_mdt693(piezo_itf, piezo_cfg):
    min_volt, max_volt = None, None
    if piezo_cfg.voltage.voltage_min.flag:
        min_volt = piezo_cfg.voltage.voltage_min.val
    if piezo_cfg.voltage.voltage_max.flag:
        max_volt = piezo_cfg.voltage.voltage_max.val
    piezo_itf.set_voltage_range(
        min_volt=min_volt, max_volt=max_volt,
        channel=piezo_cfg.device.device_id.val
    )
    piezo_cfg.set_all_flags(False)


###############################################################################
# Control
###############################################################################

# ++++++++++ MDT 693 +++++++++++++++++++++++++++++


def _set_voltage_mdt693(piezo_itf, piezo_cfg, voltage):
    piezo_itf.set_voltage(voltage, channel=piezo_cfg.device.device_id.val)


def _get_voltage_mdt693(piezo_itf, piezo_cfg):
    return piezo_itf.get_voltage(channel=piezo_cfg.device.device_id.val)
