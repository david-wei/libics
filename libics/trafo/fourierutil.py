# Package Imports
try:
    from . import addpath   # noqa
except(ImportError):
    import addpath          # noqa
import data.matrixdata as mdata


###############################################################################


def ft_pquants(pq_val, *pq_var):
    """
    Inverts and normalizes the Fourier transformed units.

    Parameters
    ----------
    pq_val : data.types.pquant
        Physical quantity function value.
    *pq_var : data.types.pquant
        Physical quantity independent variable.

    Returns
    -------
    pq_val : data.types.pquant
        Physical quantity function value.
    pq_var : tuple(data.types.pquant)
        Physical quantity independent variable.
    """
    pq_val.divide(*pq_var)
    for pq in pq_var:
        pq.invert()
    return pq_val, pq_var


def ft_matrixrect(mrect):
    """
    Calculates the Fourier transformed coordinates of a
    `data.matrixdata.MatrixRect`.

    Parameters
    ----------
    mrect : data.matrixdata.MatrixRect
        Matrix rectangle before Fourier transformation.

    Returns
    -------
    ft_mrect : data.matrixdata.MatrixRect
        Matrix rectangle after Fourier transformation.
    """
    # Calculate FT pixel sizes
    pxsize_x, pxsize_y = mrect.get_pxsize()
    pxcount_x, pxcount_y = mrect.get_pxcount()
    f_nyq_x, f_nyq_y = 1 / 2 / pxsize_x, 1 / 2 / pxsize_y
    f_cor_x, f_cor_y = 0.0, 0.0
    # Even point correction (acc. to FFT implementation)
    if pxcount_x % 2 == 0:
        f_cor_x = f_nyq_x / (pxcount_x / 2)
    if pxcount_y % 2 == 0:
        f_cor_y = f_nyq_y / (pxcount_y / 2)
    # Create FT matrix rectangle
    ft_mrect = mdata.MatrixRect()
    ft_mrect.ind_bound_x = mrect.ind_bound_x
    ft_mrect.ind_bound_y = mrect.ind_bound_y
    ft_mrect.pquant_bound_x = [-f_nyq_x, f_nyq_x - f_cor_x]
    ft_mrect.pquant_bound_y = [-f_nyq_y, f_nyq_y - f_cor_y]
    return ft_mrect
