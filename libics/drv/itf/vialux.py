import os

from libics import env
import libics.drv.itf.api.alpV42 as alp42


###############################################################################
# Initialization
###############################################################################


# Global variable for Vialux ALP4.2 API object
_ALP42 = None


def startup_alp42():
    """
    Initializes the Vialux ALP4.2 C API.

    The Vialux ALP4.2 API requires a startup and a shutdown call.
    This function checks whether startup has already been called.

    Returns
    -------
    _ALP42 : alpV42.PY_ALP_API
        Vialux ALP4.2 API object.
    """
    global _ALP42
    if _ALP42 is None:
        _ALP42 = alp42.PY_ALP_API(
            dllPath=os.path.join(env.DIR_PKG_ITFAPI, "alpV42.dll")
        )
    return _ALP42
