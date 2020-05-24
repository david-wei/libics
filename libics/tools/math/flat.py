import numpy as np


###############################################################################
# Oscillating Functions
###############################################################################


def cosine_2d(
    var, amplitude_x, amplitude_y, period_x, period_y, phase_x, phase_y,
    offset=0.0
):
    return (
        amplitude_x * np.cos(2 * np.pi * var[0] / period_x + phase_x)
        + amplitude_y * np.cos(2 * np.pi * var[1] / period_y + phase_y)
        + offset
    )
