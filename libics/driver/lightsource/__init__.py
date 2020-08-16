from .base import *
from . import ipg


###############################################################################


IPG = "IPG"
IPG_YLR = "IPG_YLR"


###############################################################################


MANUFACTURER = {
    # IPG
    IPG_YLR: IPG,
}


def get_manufacturer(model):
    return MANUFACTURER[model]


DRIVER = {
    # IPG
    IPG_YLR: ipg.IpgYLR,
}


def get_driver(model):
    return DRIVER[model]
