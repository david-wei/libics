from .base import *
from . import vialux


###############################################################################


VIALUX = "VIALUX"
VIALUX_DLP7000 = "VIALUX_DLP7000"


###############################################################################


MANUFACTURER = {
    # VIALUX
    VIALUX_DLP7000: VIALUX,
}


def get_manufacturer(model):
    return MANUFACTURER[model]


DRIVER = {
    # VIALUX
    VIALUX_DLP7000: vialux.VialuxDLP7000,
}


def get_driver(model):
    return DRIVER[model]
