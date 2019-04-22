import numpy as np
import scipy.special


###############################################################################
# Miscellaneous distribution functions
###############################################################################


def gamma_distribution_1d(x,
                          amplitude, mean, number, offset=0.0):
    r"""
    Gamma distribution in one dimension.

    .. math::
        A \frac{(N / \mu)^N}{\Gamma (N)} x^{N - 1}
        e^{-\frac{N}{\mu} x} + C

    Parameters
    ----------
    x : `float`
        Variable :math:`x`.
    amplitude : `float`
        Amplitude :math:`A`.
    mean : `float`
        Mean :math:`\mu`.
    number : `float`
        Number :math:`N`.
    offset : `float`, optional (default: 0)
        Offset :math:`C`.
    """
    exponent = (
        number * (np.log(number) - np.log(mean) - x / mean)
        + (number - 1) * np.log(x)
        - scipy.special.gammaln(number)
    )
    return amplitude * np.exp(exponent) + offset
