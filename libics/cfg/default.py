"""
Contains various default values used within the package.
"""


###############################################################################


class DATA:

    class IMAGE:

        # Pixel physical quantity, default: pixel position
        PIXEL_PQUANT = ["position x", "position y"]
        # Pixel units [x, y], default: µm
        PIXEL_UNIT = ["µm", "µm"]
        # Pixel physical quantity data type, default: float64
        PIXEL_DATATYPE = ["float64", "float64"]
        # Pixel size [x, y] with above unit, default: Manta G-145 NIR
        PIXEL_SIZE = [6.45, 6.45]
        # Pixel offset [x, y] with above unit, default: 0
        PIXEL_OFFSET = [0.0, 0.0]

        # Value physical quantity, default: intensity
        VALUE_PQUANT = "intensity"
        # Value unit, default: ADC-converted value
        VALUE_UNIT = "ADC"
        # Value physical quantity data type, default: float64
        VALUE_DATATYPE = "float64"
