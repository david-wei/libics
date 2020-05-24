# System Imports
import copy
import numpy as np
import PIL
import os

try:
    import sif_reader
except ImportError:
    print("""\
    Could not load SIF reader.
        If you are loading singularity image format files, install the
        Python package `sif_reader`.
    """)

# Package Imports
from libics.util import misc

# Subpackage Imports
from libics.file import imageutil


###############################################################################
# Bitmap (bmp)
###############################################################################


def load_bmp_to_arraydata(file_path, arraydata=None):
    """
    Reads a bitmap (bmp) file and loads the data as grayscale image into a
    `data.arraydata.ArrayData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the bitmap image file.
    arraydata : `data.arraydata.ArrayData` or `None
        Sets the array data to the loaded bitmap values.
        If `None`, creates a new ArrayData object using the
        default values as defined in `cfg.default`.

    Returns
    -------
    arraydata : `data.arraydata.ArrayData`
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
    if arraydata is None:
        arraydata = imageutil.create_default_arraydata()
    # Load bitmap
    image = np.array(PIL.Image.open(file_path).convert("L"))
    arraydata.data = image.T
    return arraydata


###############################################################################
# Portable Network Graphic (png)
###############################################################################


def load_png_to_arraydata(file_path, arraydata=None):
    """
    Reads a portable network graphic (png) file and loads the data as
    grayscale image into a `data.arraydata.ArrayData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the bitmap image file.
    arraydata : `data.arraydata.ArrayData` or `None
        Sets the array data to the loaded bitmap values.
        If `None`, creates a new ArrayData object using the
        default values as defined in `cfg.default`.

    Returns
    -------
    arraydata : `data.arraydata.ArrayData`
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
    if arraydata is None:
        arraydata = imageutil.create_default_arraydata()
    # Load bitmap
    image = np.array(PIL.Image.open(file_path).convert("L"))
    arraydata.data = image.T
    return arraydata


###############################################################################
# WinCamD (wct)
###############################################################################


def load_wct_to_arraydata(file_path, arraydata=None):
    """
    Reads a WinCamD text (wct) file and loads the data as grayscale image into
    a `data.arraydata.ArrayData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the WinCamD text file.
    arraydata : `data.arraydata.ArrayData` or `None
        Sets the array data to the loaded WinCamD values.
        If `None`, creates a new ArrayData object using the
        default values as defined in `cfg.default`.
        When available, the wct header data overwrites
        the current metadata.

    Returns
    -------
    arraydata : `data.arraydata.ArrayData`
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
    if arraydata is None:
        arraydata = imageutil.create_default_arraydata()
    # Load WinCamD text file
    image, settings = imageutil.parse_wct_to_numpy_array(file_path)
    arraydata.data = image
    arraydata.scale.scale[0] = settings["pxsize_x"]
    arraydata.scale.scale[1] = settings["pxsize_y"]
    arraydata.set_max()
    return arraydata


###############################################################################
# Singularity image format (sif)
###############################################################################


def load_sif_to_arraydata(file_path, arraydata=None):
    """
    Reads a bitmap (bmp) file and loads the data as grayscale image into a
    `data.arraydata.ArrayData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the bitmap image file.
    arraydata : `data.arraydata.ArrayData` or `None
        Sets the array data to the loaded bitmap values.
        If `None`, creates a new ArrayData object using the
        default values as defined in `cfg.default`.

    Returns
    -------
    arraydata : list(data.arraydata.ArrayData)
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
    if arraydata is None:
        arraydata = imageutil.create_default_arraydata()
    # Load sif
    images = sif_reader.np_open(file_path)[0]
    ads = []
    for im in images:
        ad = copy.deepcopy(arraydata)
        ad.data = im.T
        ads.append(ad)
    return ads
