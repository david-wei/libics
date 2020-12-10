import os
import pandas as pd

from libics.core.data.arrays import ArrayData


###############################################################################
# Utility Functions
###############################################################################


def create_default_arraydata():
    """
    Creates an instance of `data.arrays.ArrayData`.

    Returns
    -------
    ad : `data.arrays.ArrayData`
        Image default initialized ArrayData.
    """
    ad = ArrayData()
    ad.add_dim(
        {
            "offset": 0, "step": 1,
            "name": "position", "symbol": "x", "unit": "px"
        },
        {
            "offset": 0, "step": 1,
            "name": "position", "symbol": "y", "unit": "px"
        },
    )
    ad.set_data_quantity(
        name="intensity", symbol="I", unit="ADC"
    )
    return ad


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
                raise AttributeError("invalid .wct file header ({:s})"
                                     .format(file_path))
    # Read image data
    try:
        parsed_data = pd.read_csv(
            file_path, sep=" {0,2}[,;] {0,2}",
            header=None, dtype=None, skiprows=5,
            engine="python").dropna(axis=1).values
    except(pd.EmptyDataError, pd.ParserError) as e:
        raise AttributeError(e)
    # Verify pixel count
    header_shape = (header_data["pxcount_x"], header_data["pxcount_y"])
    if parsed_data.shape != header_shape:
        # Handle bugs in DataRay
        if (
            parsed_data.shape[0] == header_shape[0] - 1
            and parsed_data.shape[1] == header_shape[1]
        ):
            header_data["pxcount_x"] = header_data["pxcount_x"] - 1
        elif (
            parsed_data.shape[0] == header_shape[1]
            and parsed_data.shape[1] == header_shape[0]
        ):
            header_data["pxcount_x"], header_data["pxcount_y"] = (
                header_data["pxcount_y"], header_data["pxcount_x"]
            )
        elif (
            parsed_data.shape[0] == header_shape[1] - 1
            and parsed_data.shape[1] == header_shape[0]
        ):
            header_data["pxcount_x"], header_data["pxcount_y"] = (
                header_data["pxcount_y"] - 1, header_data["pxcount_x"]
            )
        else:
            raise AttributeError(
                f"invalid .wct pixel count "
                f"(header: ({header_data['pxcount_x']}, "
                f"{header_data['pxcount_y']}), shape: {parsed_data.shape}) "
                f"for file {file_path})"
            )
    # Remove redundant metadata keys
    header_data.pop("code_wct")
    header_data.pop("pxcount_x")
    header_data.pop("pxcount_y")
    return parsed_data, header_data
