import inspect
import json
import os
import shutil
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


def ASSUME_FILE(*args, copy_file=None):
    file_path = os.path.join(*args)
    dir_path = os.path.dirname(file_path)
    ASSUME_DIR(dir_path)
    if os.path.exists(file_path):
        if not os.path.isfile(file_path):
            raise FileExistsError("file path leads to directory")
    else:
        if copy_file is None:
            open(file_path, "w").close()
        else:
            shutil.copyfile(copy_file, file_path)
    return file_path


def READ_JSON(file_path, obj=None):
    if obj is None:
        obj = {}
    if os.path.getsize(file_path) > 0:
        with open(file_path, "r") as f:
            obj.update(json.load(f))
    return obj


def WRITE_JSON(file_path, obj):
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4)


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


# Current working directory
DIR_CWD = os.getcwd()

# LibICS source code
DIR_SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())
)))
DIR_PKGROOT = os.path.dirname(DIR_SRCROOT)
DIR_ASSETSROOT = os.path.join(DIR_PKGROOT, "assets")

# User environment
DIR_USER = (os.environ["USERPROFILE"] if sys.platform == "win32"
            else os.path.expanduser("~"))
DIR_HOME = os.path.expanduser("~")
DIR_DOCUMENTS = os.path.join(DIR_USER, "Documents")
DIR_DESKTOP = os.path.join(DIR_USER, "Desktop")
DIR_DOWNLOAD = os.path.join(DIR_USER, "Download")


###############################################################################
# LibICS directory environment
###############################################################################


DIR_LIBICS = ASSUME_DIR(DIR_HOME, ".libics")
FILE_DIRS = ASSUME_FILE(DIR_LIBICS, "env.dirs.json")
DIRS = READ_JSON(FILE_DIRS)
FILE_CONFIG = ASSUME_FILE(DIR_LIBICS, "env.config.json")
CONFIG = READ_JSON(FILE_CONFIG)


###############################################################################
# External libraries
###############################################################################


DIR_MPL = ""
FILE_MPL_STYLE = ""
FILE_MPL_STYLE_ASSET = os.path.join(DIR_ASSETSROOT, "env", "libics.mplstyle")
try:
    import matplotlib as mpl
    DIR_MPL = mpl.get_configdir()
    FILE_MPL_STYLE = ASSUME_FILE(
        DIR_MPL, "libics.mplstyle", copy_file=FILE_MPL_STYLE_ASSET
    )
except ImportError:
    pass
