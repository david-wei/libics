# System Imports
import os
import pandas as pd

# Package Imports
from libics.cfg import default as DEF
from libics.cfg import err as ERR
from libics.data import arraydata


###############################################################################
# Utility Functions
###############################################################################


def create_default_arraydata():
    """
    Creates an instance of `data.matrixdata.MatrixData` with the metadata as
    defined in `cfg.default`.

    Returns
    -------
    array_data : `data.arraydata.ArrayData`
        Image default initialized ArrayData.
    """
    array_data = arraydata.ArrayData()
    array_data.add_dim(
        offset=0,
        scale=1,
        name="position",
        symbol="x",
        unit="px"
    )
    array_data.add_dim(
        offset=0,
        scale=1,
        name="position",
        symbol="y",
        unit="px"
    )
    array_data.add_dim(
        name=DEF.DATA.IMAGE.VALUE_PQUANT,
        symbol="I",
        unit=DEF.DATA.IMAGE.VALUE_UNIT
    )
    return array_data


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
