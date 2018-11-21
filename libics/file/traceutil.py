import os

import numpy as np

from libics.data import types
from libics.util import misc


###############################################################################
# Oscilloscope
###############################################################################


def parse_tds2000_to_numpy_array(file_path, delim=","):
    """
    Parses the TDS2000 text (csv) file into a numpy array. Also reads the
    parameters as defined by the TDS2000 format.

    Parameters
    ----------
    file_path : `str`
        File path to the TDS2000 file.
    delim : `str`, optional
        Delimiter in the csv file.

    Returns
    -------
    x_data : `numpy.ndarray`
        Parsed time data.
    y_data : `numpy.ndarray`
        Parsed voltage data.
    header_data : `dict`
        The metadata included in the TDS2000 file. The keys
        include:
        `"Horizontal Units"`, `"Vertical Units"`: `str`
            x, y units.

    Raises
    ------
    FileNotFoundError
        If file path does not exist.
    """
    header_data_types = {
        "Vertical Units": str,
        "Horizontal Units": str
    }
    if not os.path.exists(file_path):
        raise FileNotFoundError(str(file_path))
    header_data = {}
    x = []
    y = []
    with open(file_path, "r") as f:
        for line in f:
            key, val, _, x_item, y_item = [item.strip(" \t\n\r")
                                           for item in line.split(delim)[:5]]
            if key != "":
                if key in header_data_types.keys():
                    try:
                        header_data[key] = header_data_types[key](val)
                    except(TypeError):
                        print("Warning: Could not typecast {:s} {:s}"
                              .format(key, str(val)))
            if x_item != "" and y_item != "":
                try:
                    x_item = float(x_item)
                    y_item = float(y_item)
                    x.append(x_item)
                    y.append(y_item)
                except(TypeError):
                    print("Warning: Could not typecast {:s} or {:s} to float"
                          .format(str(x_item), str(y_item)))
            else:
                print("Warning: Invalid value pair")
    return np.array(x), np.array(y), header_data


###############################################################################
# Spectrum Analyzer
###############################################################################


def parse_oceanoptics_to_numpy_array(file_path, delim="\t", meta_delim=":"):
    """
    Parses the OceanOptics SpectraSuite text (csv) file.

    Parameters
    ----------
    file_path : `str`
        File path to the OceanOptics file.
    delim : `str`, optional
        Data point delimiter in the csv file.
    meta_delim : `str`, optional
        Header data delimiter.

    Returns
    -------
    x_data : `numpy.ndarray`
        Parsed wavelength data.
    y_data : `numpy.ndarray`
        Parsed intensity data.
    metadata : `dict`
        The metadata included in the OceanOptics file.

    Raises
    ------
    FileNotFoundError
        If file path does not exist.
    AttributeError
        If file is corrupt.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(str(file_path))
    print("Parsing {:s}".format(str(file_path)))
    metadata = {}
    wavelengths = []
    intensities = []
    is_data = False
    with open(file_path, "r") as f:
        if (f.readline().strip(" \t\n\r") != "SpectraSuite Data File" or
                f.readline().strip(" \t\n\r") != 36 * "+"):
            raise AttributeError("Invalid spectrum file")
        for line in f:
            line = line.strip(" \t\n\r")
            if not is_data:
                if line == ">>>>>Begin Processed Spectral Data<<<<<":
                    is_data = True
                elif line != "":
                    ls = [item.strip(" \t\n\r")
                          for item in line.split(meta_delim)]
                    key, val = ls[0], ":".join(ls[1:])
                    metadata[key] = val
            else:   # is_data
                if line == ">>>>>End Processed Spectral Data<<<<<":
                    is_data = False
                elif line != "":
                    try:
                        wl, i = [float(item.strip(" \t\n\r"))
                                 for item in line.split(delim)[:2]]
                        wavelengths.append(wl * 1e-9)
                        intensities.append(i)
                    except(TypeError):
                        print("Warning: Could not cast {:s} or {:s} to float"
                              .format(str(wl), str(i)))
    x_data = np.array(wavelengths)
    y_data = np.array(intensities)
    return x_data, y_data, metadata


def parse_csv_span_agilent_to_numpy_array(
    file_path, delim=",", meta_delim=":"
):
    """
    Parses the Agilent spectrum analyzer text (csv) file.

    Parameters
    ----------
    file_path : `str`
        File path to the OceanOptics file.
    delim : `str`, optional
        Data point delimiter in the csv file.
    meta_delim : `str`, optional
        Header data delimiter.

    Returns
    -------
    frequencies : `numpy.ndarray`
        Parsed frequency data.
    amplitudes : `numpy.ndarray`
        Parsed amplitude (power) data.
    metadata : `dict`
        The metadata included in the file.
        Maps strings to data.types.ValQuantity.
        Has "_frequency_unit", "_amplitude_unit" as additional
        items that map to data.types.Quantity describing the
        units of frequency and amplitude.

    Raises
    ------
    FileNotFoundError
        If file path does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(str(file_path))
    metadata = {}
    frequencies = []
    amplitudes = []
    IS_HEADER, IS_DATA = 0, 1
    status = IS_HEADER
    # Spelling of "Frequecncy" is actual typo in spectrum analyzer
    DATA_HEADER = ["No.", "Frequecncy", "Amplitude"]
    with open(file_path, "r") as f:
        for line in f.readlines():
            # Read header
            if status == IS_HEADER:
                line = misc.split_strip(
                    line, delim=delim, strip=(" \t\n\r" + meta_delim)
                )
                if line == DATA_HEADER:
                    status = IS_DATA
                elif len(line) >= 2:
                    name = line[0]
                    val = line[1]
                    unit = None
                    if len(line) == 3:
                        unit = line[2]
                    metadata[name] = types.ValQuantity(
                        name=name, val=val, unit=unit
                    )
            # Read data
            elif status == IS_DATA:
                line = misc.split_strip(
                    line, delim=delim, strip=" \t\n\r"
                )
                if len(line) == 3:
                    freq, amp = line[1], line[2]
                    freq, freq_unit = misc.split_unit(freq)
                    amp, amp_unit = misc.split_unit(amp)
                    metadata["_frequency_unit"] = types.Quantity(
                        name="frequency", symbol="f", unit=freq_unit
                    )
                    metadata["_amplitude_unit"] = types.Quantity(
                        name="power", symbol="P", unit=amp_unit
                    )
                    frequencies.append(freq)
                    amplitudes.append(amp)
    frequencies = np.array(frequencies)
    amplitudes = np.array(amplitudes)
    return frequencies, amplitudes, metadata
