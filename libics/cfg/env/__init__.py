# System Imports
import inspect
import os


###############################################################################
# Package metadata
###############################################################################


LIBICS_VERSION_MAJOR = 0
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


DIR_SRCROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())
))))
DIR_PKGROOT = os.path.dirname(DIR_SRCROOT)
DIR_USER = os.environ["USERPROFILE"]
DIR_DOCUMENTS = os.path.join(DIR_USER, "Documents")
DIR_DOC_LIBICS = os.path.join(DIR_DOCUMENTS, "libics")
if not os.path.exists(DIR_DOC_LIBICS):
    os.makedirs(DIR_DOC_LIBICS)

###############################################################################
# File format
###############################################################################


FORMAT_JSON_INDENT = 4


###############################################################################
# Threading
###############################################################################


THREAD_DELAY_QTSIGNAL = 0.1
THREAD_DELAY_COM = 0.05
