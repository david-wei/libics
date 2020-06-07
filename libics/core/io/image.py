# System Imports
import copy
import numpy as np
import os
import PIL

from libics.core.env import logging
from libics.core.io import imageutil
from libics.core.util import misc


LOGGER = logging.get_logger("libics.core.io.image")

try:
    import sif_reader
except ImportError:
    LOGGER.info(
        "Could not load SIF reader. "
        + "If you are loading singularity image format files, "
        + "install the Python package `sif_reader`."
    )


###############################################################################
# Bitmap (bmp)
###############################################################################


def load_bmp_to_arraydata(file_path, ad=None):
    """
    Reads a bitmap (bmp) file and loads the data as grayscale image into a
    `data.arrays.ArrayData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the bitmap image file.
    ad : `data.arrays.ArrayData` or `None`
        Sets the array data to the loaded bitmap values.
        If `None`, creates a new ArrayData object using the
        default values as defined in `cfg.default`.

    Returns
    -------
    ad : `data.arrays.ArrayData`
        Image grayscales as ArrayData.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    AttributeError
        If given file is not a bitmap file.
    """
    # Check file (path)
    file_path = misc.assume_endswith(file_path, ".bmp")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    # Setup arraydata
    if ad is None:
        ad = imageutil.create_default_arraydata()
    # Load bitmap
    image = np.array(PIL.Image.open(file_path).convert("L"))
    ad.data = image.T
    return ad


###############################################################################
# Portable Network Graphic (png)
###############################################################################


def load_png_to_arraydata(file_path, ad=None):
    """
    Reads a portable network graphic (png) file and loads the data as
    grayscale image into a `data.arrays.ArrayData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the bitmap image file.
    ad : `data.arrays.ArrayData` or `None
        Sets the array data to the loaded bitmap values.
        If `None`, creates a new ArrayData object using the
        default values as defined in `cfg.default`.

    Returns
    -------
    ad : `data.arrays.ArrayData`
        Image grayscales as ArrayData.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    AttributeError
        If given file is not a bitmap file.
    """
    # Check file (path)
    file_path = misc.assume_endswith(file_path, ".png")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    # Setup arraydata
    if ad is None:
        ad = imageutil.create_default_arraydata()
    # Load bitmap
    image = np.array(PIL.Image.open(file_path).convert("L"))
    ad.data = image.T
    return ad


###############################################################################
# WinCamD (wct)
###############################################################################


def load_wct_to_arraydata(file_path, ad=None):
    """
    Reads a WinCamD text (wct) file and loads the data as grayscale image into
    a `data.arrays.ArrayData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the WinCamD text file.
    ad : `data.arrays.ArrayData` or `None
        Sets the array data to the loaded WinCamD values.
        If `None`, creates a new ArrayData object using the
        default values as defined in `cfg.default`.
        When available, the wct header data overwrites
        the current metadata.

    Returns
    -------
    ad : `data.arrays.ArrayData`
        Image grayscales as ArrayData.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    AttributeError
        If the wct file is corrupt.
    """
    # Check file (path)
    file_path = misc.assume_endswith(file_path, ".wct")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    # Setup arraydata
    if ad is None:
        ad = imageutil.create_default_arraydata()
    # Load WinCamD text file
    image, settings = imageutil.parse_wct_to_numpy_array(file_path)
    ad.data = image
    ad.set_dim(0, offset=0, step=settings["pxsize_x"])
    ad.set_dim(1, offset=0, step=settings["pxsize_y"])
    ad.var_quantity[0].unit = "µm"
    ad.var_quantity[1].unit = "µm"
    return ad


###############################################################################
# Singularity image format (sif)
###############################################################################


def load_sif_to_arraydata(file_path, ad=None):
    """
    Reads a bitmap (bmp) file and loads the data as grayscale image into a
    `data.arraydata.ArrayData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the bitmap image file.
    ad : `data.arrays.ArrayData` or `None
        Sets the array data to the loaded bitmap values.
        If `None`, creates a new ArrayData object using the
        default values as defined in `cfg.default`.

    Returns
    -------
    ads : `list(data.arrays.ArrayData)`
        Image grayscales as list of ArrayData objects.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    AttributeError
        If given file is not a bitmap file.
    """
    # Check file (path)
    file_path = misc.assume_endswith(file_path, ".sif")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    # Setup arraydata
    if ad is None:
        ad = imageutil.create_default_arraydata()
    # Load sif
    images = sif_reader.np_open(file_path)[0]
    ads = []
    for im in images:
        _ad = copy.deepcopy(ad)
        _ad.data = im.T
        ads.append(_ad)
    return ads
