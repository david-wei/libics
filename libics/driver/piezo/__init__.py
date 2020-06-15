from .base import *
from . import newport
from . import thorlabs


###############################################################################


NEWPORT = "NEWPORT"
NEWPORT_8742 = "NEWPORT_8742"

THORLABS = "THORLABS"
THORLABS_MDT69X = "THORLABS_MDT69X"
THORLABS_MDT693 = "THORLABS_MDT693"
THORLABS_MDT694 = "THORLABS_MDT694"


###############################################################################


MANUFACTURER = {
    # NEWPORT
    NEWPORT_8742: NEWPORT,
    # THORLABS
    THORLABS_MDT69X: THORLABS,
    THORLABS_MDT693: THORLABS,
    THORLABS_MDT694: THORLABS,
}


def get_manufacturer(model):
    return MANUFACTURER[model]


DRIVER = {
    # NEWPORT
    NEWPORT_8742: newport.Newport8742,
    # THORLABS
    THORLABS_MDT69X: thorlabs.ThorlabsMDT69X,
    THORLABS_MDT693: thorlabs.ThorlabsMDT69X,
    THORLABS_MDT694: thorlabs.ThorlabsMDT69X,
}


def get_driver(model):
    return DRIVER[model]
