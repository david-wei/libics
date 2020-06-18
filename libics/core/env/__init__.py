import inspect
import json
import os
import sys

from . import logging
from . import system


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


LIBICS_VERSION_MAJOR = 1
LIBICS_VERSION_MINOR = 0
LIBICS_VERSION_DEV = "dev"
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
DIR_USER = (os.environ["USERPROFILE"] if sys.platform == "win32"
            else os.path.expanduser("~"))
DIR_DOCUMENTS = os.path.join(DIR_USER, "Documents")
DIR_DOC_LIBICS = ASSUME_DIR(DIR_DOCUMENTS, "libics")


def WRITE_DIRS(dirs):
    json.dump(
        DIRS, open(os.path.join(DIR_DOC_LIBICS, "dirs.env.libics"), "w"),
        indent=4
    )


DIRS = {}
if os.path.exists(os.path.join(DIR_DOC_LIBICS, "dirs.env.libics")):
    DIRS = json.load(
        open(os.path.join(DIR_DOC_LIBICS, "dirs.env.libics"), "r")
    )
else:
    WRITE_DIRS(DIRS)


###############################################################################
# Files
###############################################################################


FILE_MPLRC = os.path.join(DIR_DOC_LIBICS, "mplrc.env.libics")
