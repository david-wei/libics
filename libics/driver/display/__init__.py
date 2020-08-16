from .base import *
from . import vialux


###############################################################################


VIALUX = "VIALUX"
VIALUX_DLP = "VIALUX_DLP"
VIALUX_V7000 = "VIALUX_V7000"


###############################################################################


MANUFACTURER = {
    # VIALUX
    VIALUX_DLP: VIALUX,
    VIALUX_V7000: VIALUX,
}


def get_manufacturer(model):
    return MANUFACTURER[model]


DRIVER = {
    # VIALUX
    VIALUX_DLP: vialux.VialuxDLP,
    VIALUX_V7000: vialux.VialuxDLP,
}


def get_driver(model):
    return DRIVER[model]
