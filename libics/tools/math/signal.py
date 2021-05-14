import numpy as np
from scipy import signal

from libics.core.data.arrays import ArrayData


###############################################################################


def correlate_g2(
    d1, d2, connected=False, normalized=False, mode="valid", method="auto"
):
    """
    Calculates a multidimensional two-point cross-correlation.

    Parameters
    ----------
    d1, d2 : `Arraylike`
        Input arrays.
    connected : `bool`
        Whether to calculate connected correlation (E[x,y]-E[x]E[y]).
    norm : `bool`
        Whether to calculate normalized correlation (E[x,y]/E[x]E[y]).
    mode : `str`
        `"valid", "full", "same"`.
    method : `str`
        `"direct", "fft", "auto"`.

    Returns
    -------
    dc : `Arraylike`
        Correlation array of the same data type as the input arrays.
    """
    # Parse data
    is_ad1 = isinstance(d1, ArrayData)
    is_ad2 = isinstance(d2, ArrayData)
    ar1 = d1.data if is_ad1 else np.array(d1)
    ar2 = d2.data if is_ad2 else np.array(d2)
    ndim = ar1.ndim
    if ndim != ar2.ndim:
        raise ValueError("data dimensions do not agree")
    # Calculate correlation
    arc = signal.correlate(ar1, ar2, mode=mode, method=method)
    arc /= np.prod(
        np.max([np.array(ar1.shape) - 1, np.array(ar2.shape) - 1], axis=0)
    )
    if connected or normalized:
        # Calculate mean
        arm = ar1.mean() * ar2.mean()
        if connected:
            arc = arc - arm
        if normalized:
            res = np.zeros_like(arc, dtype=float)
            np.divide(arc, arm, out=res, where=(arm != 0))
            arc = res
    # Calculate metadata
    if is_ad1 and is_ad2:
        dc = ArrayData(arc)
        # Set data quantity
        d1q, d2q = d1.data_quantity, d2.data_quantity
        if d1q == d2q:
            qn = "correlation"
            if d1q.name != "N/A":
                qn = f"{d1q.name} {qn}"
            if d1q.unit is None:
                qu = None
            else:
                qu = f"{d1q.unit}²"
        else:
            qn = f"{d1q.name}-{d2q.name} correlation"
            qu = None
        if d1q.symbol is None or d2q.symbol is None:
            qs = None
        else:
            qs = f"\\left< {d1q.symbol}_{{i}} {d2q.symbol}_{{i+d}} \\right>"
            mean_s = (
                f"\\left< {d1q.symbol}_i \\right> "
                f"\\left< {d2q.symbol}_i \\right>"
            )
            if np.logical_xor(connected, normalized):
                qs += " - " if connected else " / "
                qs += mean_s
            elif connected and normalized:
                qs = f"({qs} / {mean_s}) - 1"
        dc.set_data_quantity(name=qn, symbol=qs, unit=qu)
        # Set var quantity
        v1q, v2q = d1.var_quantity, d2.var_quantity
        for dim in range(dc.ndim):
            if v1q[dim] == v2q[dim]:
                dc.set_var_quantity(dim, quantity=v1q[dim])
        # Set var points
        for dim in range(dc.ndim):
            offset = d1.get_offset(dim) - d2.get_offset(dim)
            step = d1.get_step(dim)
            if np.isclose(step, d2.get_step(dim)):
                dc.set_dim(dim, offset=offset, step=step)
    else:
        dc = arc
    return dc
