# System Imports
import numpy as np
import PIL
import os

# Package Imports
from libics.cfg import default as DEF
from libics.cfg import err as ERR
from libics.data import matrixdata as mdata
from libics.util import misc

# Subpackage Imports
from libics.file import imageutil


###############################################################################
# Bitmap
###############################################################################


def load_bmp_to_matrixdata(file_path, matrixdata=None):
    """
    Reads a bitmap (bmp) file and loads the data as grayscale image into a
    `data.matrixdata.MatrixData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the bitmap image file.
    matrixdata : `data.matrixdata.MatrixData` or `None
        Sets the matrix data (`val_data` attribute) to the loaded
        bitmap values. If `None`, creates a new matrix data
        object using the default values as defined in
        `cfg.default`.

    Returns
    -------
    matrixdata : `data.matrixdata.MatrixData`
        Image grayscales as matrixdata.

    Raises
    ------
    cfg.err.DTYPE_MATRIXDATA, cfg.err.INVAL_STRUCT_NUMPY_NAME
        If the parameter types are invalid.
    FileNotFoundError
        If `file_path` does not exist.
    AttributeError
        If given file is not a bitmap file.
    """
    # Check file (path)
    file_path = misc.assume_endswith(file_path, ".bmp")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    # Setup matrixdata
    if matrixdata is None:
        matrixdata = imageutil.create_default_matrixdata()
    ERR.assertion(ERR.DTYPE_MATRIXDATA,
                  type(matrixdata) == mdata.MatrixData)
    pquants = matrixdata.get_pquants()
    ERR.assertion(ERR.INVAL_STRUCT_NUMPY_NAME,
                  pquants[0].dtype == pquants[1].dtype)
    # Load bitmap and convert to numpy grayscale image
    image = np.array(
        PIL.Image.open(file_path).convert("L"),
        dtype=pquants[0].dtype
    )
    # Assign image data
    matrixdata.set_val_data(
        image,
        val_pquant=imageutil.create_default_val_pquant()
    )
    return matrixdata


###############################################################################
# WinCamD
###############################################################################


def load_wct_to_matrixdata(file_path, matrixdata=None):
    """
    Reads a WinCamD text (wct) file and loads the data as grayscale image into
    a `data.matrixdata.MatrixData` structure.

    Parameters
    ----------
    file_path : `str`
        Path to the WinCamD text file.
    matrixdata : `data.matrixdata.MatrixData` or `None
        Sets the matrix data (`val_data` attribute) to the loaded
        WinCamD values. If `None`, creates a new matrix data
        object using the default values as defined in
        `cfg.default`. When available, the wct header data
        overwrites the current metadata.

    Returns
    -------
    matrixdata : `data.matrixdata.MatrixData`
        Image grayscales as matrixdata.

    Raises
    ------
    cfg.err.DTYPE_MATRIXDATA, cfg.err.INVAL_STRUCT_NUMPY_NAME
        If the parameter types are invalid.
    FileNotFoundError
        If `file_path` does not exist.
    AttributeError
        If the wct file is corrupt.
    """
    # Check file (path)
    file_path = misc.assume_endswith(file_path, ".wct")
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    # Setup matrixdata
    if matrixdata is None:
        matrixdata = imageutil.create_default_matrixdata()
    ERR.assertion(ERR.DTYPE_MATRIXDATA,
                  type(matrixdata) == mdata.MatrixData)
    # Load WinCamD text file and convert to numpy matrix
    image, settings = imageutil.parse_wct_to_numpy_array(file_path)
    # Assign pixel position data
    pquants = matrixdata.get_pquants()
    for i in range(2):
        pquants[i].set_unit(DEF.DATA.IMAGE.PIXEL_UNIT[i])
        pquants[i].set_dtype(DEF.DATA.IMAGE.PIXEL_DATATYPE[i])
    matrixdata.set_var_data(
        pq_pxsize_x=settings["pxsize_x"], pq_pxsize_y=settings["pxsize_y"],
        pq_offset_x=DEF.DATA.IMAGE.PIXEL_OFFSET[0],
        pq_offset_y=DEF.DATA.IMAGE.PIXEL_OFFSET[1],
        var_x_pquant=pquants[0], var_y_pquant=pquants[1]
    )
    # Assign image data
    matrixdata.set_val_data(
        image,
        val_pquant=imageutil.create_default_val_pquant()
    )
