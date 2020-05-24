import inspect
import json
import os
import sys

from . import colors    # noqa


###############################################################################
# Utility functions
###############################################################################


def ASSUME_DIR(*args):
    dir_path = os.path.join(*args)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


###############################################################################
# Package metadata
###############################################################################


LIBICS_VERSION_MAJOR = 0
LIBICS_VERSION_MINOR = 1
LIBICS_VERSION_DEV = "a"
LIBICS_VERSION = (
    str(LIBICS_VERSION_MAJOR) + "."
    + str(LIBICS_VERSION_MINOR)
    + LIBICS_VERSION_DEV
)


###############################################################################
# Directories
###############################################################################


DIR_CWD = os.getcwd()
DIR_SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())
)))
DIR_PKGROOT = os.path.dirname(DIR_SRCROOT)
DIR_ITFAPI = os.path.join(DIR_SRCROOT, "drv", "itf", "api")
DIR_USER = (os.environ["USERPROFILE"] if sys.platform == "win32"
            else os.path.expanduser("~"))
DIR_DOCUMENTS = os.path.join(DIR_USER, "Documents")
DIR_DOC_LIBICS = ASSUME_DIR(DIR_DOCUMENTS, "libics")
DIR_DOC_DATA = ASSUME_DIR(DIR_DOC_LIBICS, "data")
DIRS = {}
if os.path.exists(os.path.join(DIR_DOC_LIBICS, "dirs.env.libics")):
    DIRS = json.load(
        open(os.path.join(DIR_DOC_LIBICS, "dirs.env.libics"), "r")
    )

###############################################################################
# Data directories and files
###############################################################################


DIR_CALIBRATION = ASSUME_DIR(DIR_DOC_DATA, "calibration")
FILES_CALIBRATION = os.listdir(DIR_CALIBRATION)
FILE_MPLRC = os.path.join(DIR_DOC_LIBICS, "mplrc.env.libics")


###############################################################################
# File format
###############################################################################


FORMAT_JSON_INDENT = 4


###############################################################################
# Threading
###############################################################################


THREAD_DELAY_QTSIGNAL = 0.1
THREAD_DELAY_COM = 0.05


###############################################################################
# Logging
###############################################################################


VERBOSITY_SILENT = -1
VERBOSITY_NORMAL = 0
VERBOSITY_VERBOSE = 1
VERBOSITY_VERY_VERBOSE = 2
VERBOSITY_DEBUG = 3
