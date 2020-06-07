import copy
import numpy as np

from libics.core.data.arrays import ArrayData


###############################################################################


def fft_arraydata(ad, mode=None):
    """
    Computes the Fast Fourier Transform of a `data.matrixdata.MatrixData`
    object.

    Parameters
    ----------
    ad : data.arrays.ArrayData
        The array data to be Fourier transformed.
    mode : None or str
        None: Saves full complex values of Fourier transform.
        "abs": Saves spectrum (modulus of Fourier transform).
        "phase": Saves phase of Fourier transform.
        "re": Saves real part of Fourier transform.
        "im": Saves imaginary part of Fourier transform.

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
    # Perform FFT
    ft_data = np.fft.fftshift(np.fft.fftn(ad.data))
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
        q.name = "inverse " + q.name
        if q.symbol is not None:
            q.symbol = "1 / " + q.symbol
        if q.unit is not None:
            if q.unit not in units:
                units[q.unit] = 1
            else:
                units[q.unit] += 1
            q.unit = "1 / " + q.unit
    # Set data quantity
    for unit, num in units.items():
        if num > 1:
            ft.data_quantity.unit += " {:s}^{:d}".format(unit, num)
        else:
            ft.data_quantity.unit += " {:s}".format(unit)
    return ft


###############################################################################


if __name__ == "__main__":

    # Test function
    def f(x):
        return np.cos(2 * np.pi * x) + np.sin(5 * np.pi * x)

    # ----------
    # Array data
    # ----------
    from libics.display import plotdefault, plot
    pcfg = plotdefault.get_plotcfg_arraydata_1d()
    fcfg = plotdefault.get_figurecfg()
    # Setup array data
    offset = 100
    scale = 0.05
    num = 1001
    ar = np.linspace(offset, offset + scale * (num - 1), num=num)
    ad = ArrayData()
    ad.add_dim(offset=offset, scale=scale, name="time", unit="s", symbol="t")
    ad.add_dim(name="amplitude", unit="V", symbol="A")
    ad.data = f(ar)
    fig = plot.Figure(fcfg, pcfg, data=ad)
    fig.plot()
    fig.show()
    # FFT
    ft_ad = fft_arraydata(ad, mode="abs")
    fig = plot.Figure(fcfg, pcfg, data=ft_ad)
    fig.plot()
    fig.show()
