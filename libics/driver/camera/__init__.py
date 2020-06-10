from .base import *
from . import vimba
from . import vrmagic


###############################################################################


ALLIEDVISION = "ALLIEDVISION"
ALLIEDVISION_MANTA_G145B_NIR = "ALLIEDVISION_MANTA_G145B_NIR"

VRMAGIC = "VRMAGIC"
VRMAGIC_VRMCX = "VRMAGIC_VRMCX"


###############################################################################


MANUFACTURER = {
    # ALLIEDVISION
    ALLIEDVISION_MANTA_G145B_NIR: ALLIEDVISION,
    # VRMAGIC
    VRMAGIC_VRMCX: VRMAGIC,
}


def get_manufacturer(model):
    return MANUFACTURER[model]


DRIVER = {
    # ALLIEDVISION
    ALLIEDVISION_MANTA_G145B_NIR: vimba.AlliedVisionMantaG145BNIR,
    # VRMAGIC
    VRMAGIC_VRMCX: vrmagic.VRmagicVRmCX,
}


def get_driver(model):
    return DRIVER[model]
