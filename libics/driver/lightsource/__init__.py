# System Imports
import abc
import numpy as np

# Package Imports





class LaserCfg():

    """
    DrvCfgBase -> LaserCfg.

    Parameters
    ----------
    current : float
        Current in Ampere (A).
    temperature : float
        Temperature in degrees (°C).
    """

    def __init__(
        self, current=0.1, temperature=25.0,
        cls_name="LaserCfg", ll_obj=None, **kwargs
    ):
        if "driver" not in kwargs.keys():
            kwargs["driver"] = DRV_DRIVER.LASER
        super().__init__(cls_name=cls_name, **kwargs)
        if ll_obj is not None:
            ll_obj_dict = dict(ll_obj.__dict__)
            for key in list(ll_obj_dict.keys()):
                if key.startswith("_"):
                    del ll_obj_dict[key]
            self.__dict__.update(ll_obj_dict)
        self.current = current
        self.temperature = temperature

    def get_hl_cfg(self):
        return self




###############################################################################


def get_laser_drv(cfg):
    if cfg.model == drv.DRV_MODEL.IPG_YLR:
        return IpgYLR(cfg)


class LaserDrvBase():

    """
    Laser driver API.
    """

    def __init__(self, cfg):
        assert(isinstance(cfg, drv.LaserCfg))
        super().__init__(cfg=cfg)

    @abc.abstractmethod
    def run(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def write_power(self, power):
        pass

    @abc.abstractmethod
    def read_power(self):
        pass

    @abc.abstractmethod
    def write_current(self, current):
        pass

    @abc.abstractmethod
    def read_current(self):
        pass

    # ++++ Write/read methods +++++++++++

    def _write_current(self, value):
        pass

    def _read_current(self):
        pass

    def _write_temperature(self, value):
        pass

    def _read_temperature(self):
        pass


###############################################################################


class IpgYLR(LaserDrvBase):

    def __init__(self, cfg):
        super().__init__(cfg)

    def connect(self):
        super().connect()

    def run(self):
        self.interface_access.acquire()
        self._itf_send("EMON")
        self._itf_recv_ack("EMON")
        self.interface_access.release()

    def stop(self):
        self.interface_access.acquire()
        self._itf_send("EMOFF")
        self._itf_recv_ack("EMOFF")
        self.interface_access.release()

    def write_power(self, power):
        raise NotImplementedError

    def read_power(self):
        self.interface_access.acquire()
        self._interface.send("ROP")
        recv = self._itf_recv_val("ROP")
        try:
            power = float(recv)
        except ValueError:
            power = recv
        self.interface_access.release()
        return power

    def write_current(self, current):
        self.interface_access.acquire()
        self._write_current(current)
        self.interface_access.release()

    def read_current(self):
        self.interface_access.acquire()
        current = self._read_current()
        self.interface_access.release()
        return current

    # ++++ Write/read methods +++++++++++

    def _write_current(self, value):
        self._interface.send("SDC {:f}".format(value))
        if not np.close(value, float(self._itf_recv_val("SDC"))):
            raise RuntimeError("writing current failed")

    def _read_current(self):
        self._interface.send("RCS")
        return float(self._itf_recv_val("RCS"))

    def _write_temperature(self, value):
        raise NotImplementedError

    def _read_temperature(self):
        self._interface.send("RCT")
        return float(self._itf_recv_val("RCT"))

    # ++++ Helper methods +++++++++++++++

    def _itf_send(self, msg):
        self._interface.send(msg)

    def _itf_recv(self):
        return self._strip_recv(self._interface.recv())

    def _itf_recv_ack(self, msg):
        if msg != self._itf_recv():
            raise RuntimeError("Command not acknowledged")

    def _itf_recv_val(self, msg):
        recv = self._itf_recv().split(":")
        if len(recv) <= 1 or msg != self._strip_recv(recv[0]):
            raise RuntimeError("Command not acknowledged")
        return self._strip_recv(recv[-1])

    @staticmethod
    def _strip_recv(value):
        return value.lstrip("\n\r*[ \x00").rstrip("\n\r] \x00")
