from .base import *
from . import agilent
from . import stanford
from . import tektronix
from . import yokagawa


###############################################################################


AGILENT = "AGILENT"
AGILENT_N9320X = "AGILENT_N9320X"

STANFORD = "STANFORD"
STANFORD_SR760 = "STANFORD_SR760"

TEKTRONIX = "TEKTRONIX"
TEKTRONIX_TDS100X = "TEKTRONIX_TDS100X"

YOKAGAWA = "YOKAGAWA"
YOKAGAWA_AQ6315 = "YOKAGAWA_AQ6315"


###############################################################################


MANUFACTURER = {
    # AGILENT
    AGILENT_N9320X: AGILENT,
    # STANFORD
    STANFORD_SR760: STANFORD,
    # TEKTRONIX:
    TEKTRONIX_TDS100X: TEKTRONIX,
    # YOKAGAWA
    YOKAGAWA_AQ6315: YOKAGAWA,
}


def get_manufacturer(model):
    return MANUFACTURER[model]


DRIVER = {
    # AGILENT
    AGILENT_N9320X: agilent.AgilentN9320X,
    # STANFORD
    STANFORD_SR760: stanford.StanfordSR760,
    # TEKTRONIX:
    TEKTRONIX_TDS100X: tektronix.TektronixTDS100X,
    # YOKAGAWA
    YOKAGAWA_AQ6315: yokagawa.YokagawaAQ6315,
}


def get_driver(model):
    return DRIVER[model]
