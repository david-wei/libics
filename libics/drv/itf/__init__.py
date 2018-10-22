from . import camcfg    # noqa
from . import gpib      # noqa
from . import mdt693    # noqa
from . import itf       # noqa
from . import itfbin    # noqa
from . import itftxt    # noqa
from . import vimba     # noqa


###############################################################################


def get_itf(cfg):
    if cfg.protocol == itf.ITF_PROTOCOL.TEXT:
        return itftxt.get_txt_itf(cfg)
    elif cfg.protocol == itf.ITF_PROTOCOL.BINARY:
        return itfbin.get_bin_itf(cfg)
