from .base import *
from . import prologix


###############################################################################


PROLOGIX = "PROLOGIX"
PROLOGIX_GPIB = "PROLOGIX_GPIB"


###############################################################################


MANUFACTURER = {
    # PROLOGIX
    PROLOGIX_GPIB: PROLOGIX,
}


def get_manufacturer(model):
    return MANUFACTURER[model]


DRIVER = {
    # PROLOGIX
    PROLOGIX_GPIB: prologix.PrologixGpib,
}


def get_driver(model):
    return DRIVER[model]
