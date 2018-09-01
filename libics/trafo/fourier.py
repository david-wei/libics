# System Imports
import copy
import numpy as np

# Package Imports
from libics.cfg import err as ERR
from libics.data import matrixdata as mdata
from libics.data import types

# Subpackage Imports
from libics.trafo import fourierutil


###############################################################################


def fft_matrixdata(data, mode=None):
    """
    Computes the Fast Fourier Transform of a `data.matrixdata.MatrixData`
    object.

    Parameters
    ----------
    data : data.matrixdata.MatrixData
        The matrix data to be Fourier transformed.
    mode : None or str
        None: Saves full complex values of Fourier transform.
        "abs": Saves spectrum (modulus of Fourier transform).
        "phase": Saves phase of Fourier transform.
        "re": Saves real part of Fourier transform.
        "im": Saves imaginary part of Fourier transform.

    Returns
    -------
    data : data.matrixdata.MatrixData
        Fourier transformed data.

    Raises
    ------
    cfg.err.DTYPE_MATRIXDATA, cfg.err.INVAL_SET
        If parameters are invalid.
    """
    ERR.assertion(ERR.DTYPE_MATRIXDATA,
                  type(data) == mdata.MatrixData)
    # Perform FFT
    ft_data = np.fft.fftshift(np.fft.fftn(data.get_val_data()))
    if mode is None:
        pass
    elif mode == "abs":
        ft_data = np.abs(ft_data)
    elif mode == "phase":
        ft_data = np.angle(ft_data)
    elif mode == "re":
        ft_data = np.real(ft_data)
    elif mode == "im":
        ft_data = np.imag(ft_data)
    else:
        raise ERR.INVAL_SET(str(ERR.INVAL_SET))
    # Set matrixdata
    pq_x, pq_y, pq_val = copy.deepcopy(data.get_pquants())  # pquant is mutable
    pq_val, (pq_x, pq_y) = fourierutil.ft_pquants(pq_val, pq_x, pq_y)
    ft_mrect = fourierutil.ft_matrixrect(data.var_rect)
    data.set_val_data(ft_data, val_pquant=pq_val)
    data.set_var_data(matrix_rect=ft_mrect,
                      var_x_pquant=pq_x, var_y_pquant=pq_y)
    return data


###############################################################################


if __name__ == "__main__":

    # Test function
    def f(x):
        return (
            np.cos(2 * np.pi * np.sqrt(x**2 + y**2))
            + np.sin(5 * np.pi * np.sqrt(x**2 + y**2))
        )
    # Create test data
    array = np.full((6, 6), 0.0)
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            array[x, y] = np.sqrt(x**2 + y**2)
    array = f(array)
    pq_val = types.pquant("power", "µW")
    pq_x = types.pquant("x position", "µm")
    pq_y = types.pquant("y position", "µm")
    data = mdata.MatrixData()
    data.set_val_data(array, val_pquant=pq_val)
    data.set_var_data(
        pq_pxsize_x=6.45, pq_pxsize_y=6.45, pq_offset_x=0.0, pq_offset_y=0.0,
        var_x_pquant=pq_x, var_y_pquant=pq_y
    )
    print(data)

    # FFT
    ft_data = fft_matrixdata(data, mode="abs")
    print(ft_data)
