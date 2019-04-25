# System Imports
import numpy as np

# Package Imports
from libics.data import arraydata

# Subpackage Imports
from libics.trafo import fourierutil


###############################################################################


def fft_arraydata(data, mode=None):
    """
    Computes the Fast Fourier Transform of a `data.matrixdata.MatrixData`
    object.

    Parameters
    ----------
    data : data.arraydata.ArrayData
        The array data to be Fourier transformed.
    mode : None or str
        None: Saves full complex values of Fourier transform.
        "abs": Saves spectrum (modulus of Fourier transform).
        "phase": Saves phase of Fourier transform.
        "re": Saves real part of Fourier transform.
        "im": Saves imaginary part of Fourier transform.

    Returns
    -------
    data : data.arraydata.ArrayData
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
    ft_data = np.fft.fftshift(np.fft.fftn(data.data))
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
    # Calculate array scale
    ft_scale = fourierutil.ft_arrayscale(data.scale, data.data.shape)
    # Construct Fourier transformed data
    data = arraydata.ArrayData()
    data.scale = ft_scale
    data.data = ft_data
    return data


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
    ad = arraydata.ArrayData()
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
