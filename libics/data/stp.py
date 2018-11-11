from libics import cfg


###############################################################################


class SetupCfgBase(cfg.CfgBase):

    """
    Container class for storing unstructured flat data.

    The data items are key-value pairs where the attribute name is the key and
    the attribute value is the corresponding value. Attribute values may only
    be of built-in Python types.
    """

    def __init__(self, **attrs):
        self.__dict__.update(attrs)

    def get_hl_cfg(self):
        return self
