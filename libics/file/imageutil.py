# System Imports
import os
import pandas as pd

# Package Imports
from libics.cfg import default as DEF
from libics.cfg import err as ERR
from libics.data import matrixdata as mdata
from libics.data import types


###############################################################################
# Utility Functions
###############################################################################


def create_default_matrixdata():
    """
    Creates an instance of `data.matrixdata.MatrixData` with the metadata as
    defined in `cfg.default`.

    Returns
    -------
    matrixdata : `data.matrixdata.MatrixData`
        Image default initialized matrixdata.
    """
    matrixdata = mdata.MatrixData()
    matrixdata.set_var_data(
        pq_pxsize_x=DEF.DATA.IMAGE.PIXEL_SIZE[0],
        pq_pxsize_y=DEF.DATA.IMAGE.PIXEL_SIZE[1],
        pq_offset_x=DEF.DATA.IMAGE.PIXEL_OFFSET[0],
        pq_offset_y=DEF.DATA.IMAGE.PIXEL_OFFSET[1],
        var_x_pquant=types.pquant(
            name=DEF.DATA.IMAGE.PIXEL_PQUANT[0],
            unit=DEF.DATA.IMAGE.PIXEL_UNIT[0],
            dtype=DEF.DATA.IMAGE.PIXEL_DATATYPE[0]
        ),
        var_y_pquant=types.pquant(
            name=DEF.DATA.IMAGE.PIXEL_PQUANT[1],
            unit=DEF.DATA.IMAGE.PIXEL_UNIT[1],
            dtype=DEF.DATA.IMAGE.PIXEL_DATATYPE[1]
        )
    )
    return matrixdata


def create_default_val_pquant():
    """
    Creates an physical quantity object (`data.types.pquant`) for the image
    intensity as defined in `cfg.default`.

    Returns
    -------
    val_pquant : `data.types.pquant`
        Image intensity physical quantity object.
    """
    pquant = types.pquant(
        name=DEF.DATA.IMAGE.VALUE_PQUANT,
        unit=DEF.DATA.IMAGE.VALUE_UNIT,
        dtype=DEF.DATA.IMAGE.VALUE_DATATYPE
    )
    return pquant


###############################################################################
# WinCamD
###############################################################################


def parse_wct_to_numpy_array(file_path):
    """
    Parses the WinCamD text (wct) file into a numpy array. Also reads the
    parameters as defined by the wct format.

    Parameters
    ----------
    file_path : `str`
        File path to the wct file.

    Returns
    -------
    parsed_data : `numpy.ndarray(2)`
        The wct image data as numpy matrix.
    header_data : `dict`
        The metadata included in the wct file. The keys include:
        `pxsize_x`, `pxsize_y`: `float`
            Pixel size in [x, y] direction in [µm].

    Raises
    ------
    FileNotFoundError
        If file path does not exist.
    AttributeError
        If file is corrupt.
    """
    # Check file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    # Initialize temporary variables
    parsed_data = None
    header_data = {}
    # Read file
    with open(file_path) as file:
        # Read header (5 lines)
        for _ in range(5):
            line = file.readline().split("=")
            for i in range(len(line)):
                line[i] = line[i].strip("; \n\t")
            if line[0] == "WinCamD text file":
                header_data["code_wct"] = True
            elif line[0] == "Width":
                header_data["pxcount_x"] = int(line[1])
            elif line[0] == "Height":
                header_data["pxcount_y"] = int(line[1])
            elif line[0] == "X pixels size":
                header_data["pxsize_x"] = float(line[1])
            elif line[0] == "Y pixels size":
                header_data["pxsize_y"] = float(line[1])
        # Verify header syntax
        for header_item in ["code_wct", "pxcount_x", "pxcount_y",
                            "pxsize_x", "pxsize_y"]:
            if header_item not in header_data.keys():
                raise AttributeError(ERR.INVAL_FILE_WCTHEADER)
    # Read image data
    try:
        parsed_data = pd.read_csv(
            file_path, sep=" {0,2}[,;] {0,2}",
            header=None, dtype=None, skiprows=5,
            engine="python").dropna(axis=1).values
    except(pd.EmptyDataError, pd.ParserError) as e:
        raise AttributeError(e)
    # Verify pixel count
    if (parsed_data.shape
            != (header_data["pxcount_x"], header_data["pxcount_y"])):
        raise AttributeError(ERR.INVAL_FILE_WCTPXCOUNT)
    # Remove redundant metadata keys
    header_data.pop("code_wct")
    header_data.pop("pxcount_x")
    header_data.pop("pxcount_y")
    return parsed_data, header_data
