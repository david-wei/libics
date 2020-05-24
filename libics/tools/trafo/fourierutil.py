import copy


###############################################################################


def ft_arrayscale(scale, shape):
    """
    Calculates the Fourier transformed coordinates of a
    `data.arraydata.ArrayScale`.

    Parameters
    ----------
    scale : data.arraydata.ArrayScale
        Array scale before Fourier transformation.
    shape : tuple(int)
        Shape of data.

    Returns
    -------
    scale : data.arraydata.ArrayScale
        Array scale after Fourier transformation.
    """
    scale = copy.deepcopy(scale)
    # FFT Nyquist frequency
    f_nyq = [1 / 2 / scale.scale[i] for i, _ in enumerate(shape)]
    # FFT even point correction
    f_cor = [2 * f_nyq[i] / s if s % 2 == 0 else 0
             for i, s in enumerate(shape)]
    # Set array scale scaling
    for i, s in enumerate(shape):
        scale.scale[i] = (2 * f_nyq[i] - f_cor[i]) / (s - 1)
        scale.offset[i] = -f_nyq[i]
    scale.set_max(shape)
    # Set array scale quantity
    units = {}
    for q in scale.quantity[:-1]:
        q.name = "inverse " + q.name
        if q.symbol is not None:
            q.symbol = "1 / " + q.symbol
        if q.unit is not None:
            if q.unit not in units:
                units[q.unit] = 1
            else:
                units[q.unit] += 1
            q.unit = "1 / " + q.unit
    for unit, num in units.items():
        if num > 1:
            scale.quantity[-1].unit += " {:s}^{:d}".format(unit, num)
        else:
            scale.quantity[-1].unit += " {:s}".format(unit)
    return scale
