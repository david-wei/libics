from libics import cfg
from libics.data import types
from libics.util import misc


###############################################################################


class SetupCfgBase(cfg.CfgBase):

    """
    Container class for storing unstructured flat data.

    Attributes are stored as object attributes in the object dictionary. If
    not explicitly specified, the attribute values must be of
    `data.types.ValQuantity` type.
    """

    def __init__(self, **attrs):
        for key, val in attrs.items():
            if isinstance(val, types.ValQuantity):
                pass
            elif isinstance(val, types.Quantity):
                attrs[key] = types.ValQuantity(
                    name=val.name, symbol=val.symbol, unit=val.unit
                )
            elif isinstance(val, dict):
                attrs[key] = misc.assume_construct_obj(types.ValQuantity)
            else:
                attrs[key] = types.ValQuantity(name=key, val=val)
        self.__dict__.update(attrs)

    def get_hl_cfg(self):
        return self
