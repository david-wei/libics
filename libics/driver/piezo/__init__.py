from .base import *
from . import newport
from . import thorlabs


###############################################################################


NEWPORT = "NEWPORT"
NEWPORT_8742 = "NEWPORT_8742"

THORLABS = "THORLABS"
THORLABS_MDT69XA = "THORLABS_MDT69XA"
THORLABS_MDT693A = "THORLABS_MDT693A"
THORLABS_MDT694A = "THORLABS_MDT694A"


###############################################################################


MANUFACTURER = {
    # NEWPORT
    NEWPORT_8742: NEWPORT,
    # THORLABS
    THORLABS_MDT69XA: THORLABS,
    THORLABS_MDT693A: THORLABS,
    THORLABS_MDT694A: THORLABS,
}


def get_manufacturer(model):
    return MANUFACTURER[model]


DRIVER = {
    # NEWPORT
    NEWPORT_8742: newport.Newport8742,
    # THORLABS
    THORLABS_MDT69XA: thorlabs.ThorlabsMDT69XA,
    THORLABS_MDT693A: thorlabs.ThorlabsMDT69XA,
    THORLABS_MDT694A: thorlabs.ThorlabsMDT69XA,
}


def get_driver(model):
    return DRIVER[model]
