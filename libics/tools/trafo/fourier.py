import copy
import numpy as np

from libics.core.data.arrays import ArrayData


###############################################################################


def fft(ad, mode=None, symmetric=True):
    """
    Computes the Fast Fourier Transform of an array.

    Parameters
    ----------
    ad : `np.ndarray` or `ArrayData`
        The array data to be Fourier transformed.
    mode : `None` or `str`
        `None`: Saves full complex values of Fourier transform.
        `"abs`": Saves spectrum (modulus of Fourier transform).
        `"phase"`: Saves phase of Fourier transform.
        `"re"`: Saves real part of Fourier transform.
        `"im"`: Saves imaginary part of Fourier transform.
    symmetric : `bool`
        Whether to use symmetric Fourier transform.

    Returns
    -------
    ft : data.arrays.ArrayData
        Fourier transformed data.

    Raises
    ------
    ValueError
        If parameters are invalid.

    Notes
    -----
    FFT is implemented as asymmetric transform with respect to frequency and
    not angular frequency (i.e. 2πft, not ωt).
    """
    # Parse data type
    if isinstance(ad, np.ndarray):
        ar = ad
    elif isinstance(ad, ArrayData):
        ar = ad.data
    else:
        raise ValueError(f"invalid data type {type(ad)}")
    # Perform FFT
    ft_data = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ar)))
    if symmetric is True:
        if isinstance(ad, np.ndarray):
            ft_data /= np.sqrt(np.prod(ft_data.shape))
        elif isinstance(ad, ArrayData):
            ft_data *= np.prod(ad.step)
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
        raise ValueError("invalid mode: {:s}".format(str(mode)))
    # Return data if no metadata
    if isinstance(ad, np.ndarray):
        return ft_data

    # ++++++++++++++++++++++++++
    # Calculate variable scaling
    # ++++++++++++++++++++++++++
    # Construct Fourier transformed data
    ft = copy.deepcopy(ad)
    ft.data = ft_data
    # FFT Nyquist frequency
    f_nyq = [1 / 2 / ad.get_step(i) for i, _ in enumerate(ad.shape)]
    # FFT even point correction
    f_cor = [2 * f_nyq[i] / s if s % 2 == 0 else 0
             for i, s in enumerate(ad.shape)]
    # Set variable scaling
    for i, s in enumerate(ad.shape):
        s = (2 * f_nyq[i] - f_cor[i]) / (s - 1)
        o = -f_nyq[i]
        ft.set_dim(i, offset=o, step=s)
    # Set variable quantity
    units = {}
    for q in ft.var_quantity:
        if q.name[:8] == "inverse ":
            q.name = q.name[8:]
        else:
            q.name = f"inverse {q.name}"
        if q.symbol is not None:
            if q.symbol[:4] == "1 / ":
                if q.symbol[4] == "(" and q.symbol[-1] == ")":
                    q.symbol = q.symbol[5:-1]
                else:
                    q.symbol = q.symbol[4:]
            else:
                q.symbol = f"1 / ({q.symbol})"
        if q.unit is not None:
            if q.unit not in units:
                units[q.unit] = 1
            else:
                units[q.unit] += 1
            if q.unit[:4] == "1 / ":
                if q.unit[4] == "(" and q.unit[-1] == ")":
                    q.unit = q.unit[5:-1]
                else:
                    q.unit = q.unit[4:]
            else:
                q.unit = f"1 / ({q.unit})"
    # Set data quantity
    if ft.data_quantity.unit is not None:
        for unit, num in units.items():
            if num > 1:
                ft.data_quantity.unit += " {:s}^{:d}".format(unit, num)
            else:
                ft.data_quantity.unit += f" {unit}"
    return ft
