import copy
import numpy as np
import re

from libics.core.data.arrays import ArrayData
from libics.core.util import misc


###############################################################################


def fft(data, axes=None, time_range=None, frequency_start=None):
    """
    Computes the Fast Fourier Transform of an array.

    Computes the forward (time->frequency) transformation and interpretes
    the data transformation as continuous Fourier transform.

    .. math::
        \\tilde{g} (f) = \\int_{-\\inf}^{\\inf} g (t) e^{-i 2 \\pi f t} dt

    Parameters
    ----------
    data : `Array[complex or float]`
        The array data to be Fourier transformed.
    axes : `int` or `Iter[int]`
        Indices of dimensions to be transformed.
        If `None`, transforms all.
    time_range : `tuple(float)|str|None` or `Iter[thereof]`
        Time range which `data` represents.
        If `Iter[...]`, specifies the time range per transformed axis.
        `tuple(float)`: specifies the time range `(ta, tb)`.
        `None`: If `data` is `ArrayData` infers time range, otherwise `"zero"`.
        `"zero"`: uses time range `(0, N-1)`. `N` is data size in resp. dim.
        `"center"`: uses time range `(-(N//2), (N+1)//2 - 1)`.
    frequency_start : `float|str|None` or `Iter[thereof]`
        Frequency range start of returned data.
        If `Iter[...]`, uses the respective value in each axis.
        `float`: specifies the frequency-range start value.
        `None` or `"zero"`: uses `0`.
        `"center": uses `-(N//2) * df`.

    Returns
    -------
    ft : `Array[complex]`
        Fourier transformed data. Returns same data type as `data`.

    Notes
    -----
    * Variable notation:
      data range `(ta, tb)`, data steps `dt`, data span `tt` for time domain.
      `(fa, fb), df, ff` analogously for frequency domain.
    * Setting explicit time ranges and frequency ranges are important to
      obtain correct Fourier transformed phases and correct assignment of
      frequencies.
    """
    # Parse data type
    is_ad = isinstance(data, ArrayData)
    ad = ArrayData(data)
    ar = np.array(data, dtype="complex")
    # Parse active axes
    if axes is None:
        axes = np.arange(ad.ndim)
    else:
        axes = np.array(misc.assume_iter(axes))
    # Parse time and frequency domain
    _dom = _get_fft_domains(ad, axes, time_range, frequency_start)
    t_phase_correction = _get_fft_domain_phase_correction(
        _dom["ns"], _dom["fas"]/_dom["ffs"], ad.ndim, axes
    )
    f_phase_correction = _get_fft_domain_phase_correction(
        _dom["ns"], _dom["tas"]/_dom["tts"], ad.ndim, axes,
        scalar=np.sum(_dom["fas"] * _dom["tas"])
    )
    data_scaling = np.prod(_dom["dts"])
    # Perform FFT
    if t_phase_correction is not None:
        ar *= np.exp(-1j * 2 * np.pi * t_phase_correction)
    ft_ar = data_scaling * np.fft.fftn(ar, axes=axes, norm="backward")
    if f_phase_correction is not None:
        ft_ar *= np.exp(-1j * 2 * np.pi * f_phase_correction)
    # Return data
    if not is_ad:
        return ft_ar
    else:
        # Set ArrayData
        ft_ad = ad.copy()
        ft_ad.data = ft_ar
        for i, d in enumerate(axes):
            ft_ad.set_dim(d, offset=_dom["fas"][i], step=_dom["dfs"][i])
            ft_ad.var_quantity[d] = _get_fft_var_quantity(ad.var_quantity[d])
        return ft_ad


def ifft(data, axes=None, frequency_range=None, time_start=None):
    """
    Computes the inverse Fast Fourier Transform of an array.

    Computes the backward (frequency->time) transformation and interpretes
    the data transformation as continuous Fourier transform.

    .. math::
        g (t) = \\int_{-\\inf}^{\\inf} \\tilde{g} (f) e^{i 2 \\pi f t} df

    For parameters and return values, see :py:meth:`fft`.
    """
    # Parse data type
    is_ad = isinstance(data, ArrayData)
    ad = ArrayData(data)
    ar = np.array(data, dtype="complex")
    # Parse active axes
    if axes is None:
        axes = np.arange(ad.ndim)
    else:
        axes = np.array(misc.assume_iter(axes))
    # Parse time and frequency domain
    _dom = _get_fft_domains(ad, axes, frequency_range, time_start)
    f_phase_correction = _get_fft_domain_phase_correction(
        _dom["ns"], _dom["fas"]/_dom["ffs"], ad.ndim, axes
    )
    t_phase_correction = _get_fft_domain_phase_correction(
        _dom["ns"], _dom["tas"]/_dom["tts"], ad.ndim, axes,
        scalar=np.sum(_dom["fas"] * _dom["tas"])
    )
    data_scaling = np.prod(_dom["dts"])
    # Perform FFT
    if f_phase_correction is not None:
        ar *= np.exp(1j * 2 * np.pi * f_phase_correction)
    ft_ar = data_scaling * np.fft.ifftn(ar, axes=axes, norm="forward")
    if t_phase_correction is not None:
        ft_ar *= np.exp(1j * 2 * np.pi * t_phase_correction)
    # Return data
    if not is_ad:
        return ft_ar
    else:
        # Set ArrayData
        ft_ad = ad.copy()
        ft_ad.data = ft_ar
        for i, d in enumerate(axes):
            ft_ad.set_dim(d, offset=_dom["fas"][i], step=_dom["dfs"][i])
            ft_ad.var_quantity[d] = _get_fft_var_quantity(ad.var_quantity[d])
        return ft_ad


###############################################################################
# Helper functions
###############################################################################


def _get_fft_domains(ad, axes, time_range, frequency_start):
    """
    Extracts time and frequency domain represented by data for FFT.
    """
    # Parse time domain
    ns = np.array([ad.shape[d] for d in axes])
    tas = np.array([ad.get_offset(d) for d in axes])
    dts = np.array([ad.get_step(d) for d in axes])
    tts = ns * dts
    if time_range is not None:
        # Vectorize
        is_scalar = True
        if not isinstance(time_range, str):
            try:
                _time_range_new = []
                for _tr in time_range:
                    if _tr is None or isinstance(_tr, str):
                        _time_range_new.append(_tr)
                    elif np.array(_tr).ndim == 1 and np.array(_tr).size == 2:
                        _time_range_new.append(_tr)
                    else:
                        raise TypeError
                is_scalar = False
                time_range = _time_range_new
            except TypeError:
                pass
        if is_scalar:
            time_range = len(axes) * [time_range]
        # Parse
        for i, _tr in enumerate(time_range):
            if _tr is None:
                continue
            elif isinstance(_tr, str):
                dts[i] = 1
                tts[i] = ns[i]
                if _tr.lower() == "zero":
                    tas[i] = 0
                elif _tr.lower() == "center":
                    tas[i] = -(ns[i] // 2)
                else:
                    raise ValueError(
                        f"Invalid time_range[{i:d}]: {time_range}"
                    )
            else:
                tas[i] = _tr[0]
                tts[i] = _tr[1] - _tr[0]
                dts[i] = tts[i] / (ns[i] - 1)
    tbs = tas + tts
    # Parse frequency domain
    fas = np.zeros(len(axes), dtype=float)
    ffs = 1 / dts
    dfs = 1 / tts
    if frequency_start is not None:
        # Vectorize
        is_scalar = True
        if not isinstance(frequency_start, str):
            try:
                _frequency_start_new = []
                for _fs in frequency_start:
                    if _fs is None or isinstance(_fs, str) or np.isscalar(_fs):
                        _frequency_start_new.append(_fs)
                    else:
                        raise TypeError
                is_scalar = False
                frequency_start = _frequency_start_new
            except TypeError:
                pass
        if is_scalar:
            frequency_start = len(axes) * [frequency_start]
        # Parse
        for i, _fs in enumerate(frequency_start):
            if _fs is None:
                continue
            elif isinstance(_fs, str):
                if _fs.lower() == "center":
                    fas[i] = -(ns[i] // 2) * dfs[i]
                elif _fs.lower() == "zero":
                    pass
                else:
                    raise ValueError(
                        f"Invalid frequency_start[{i:d}]: {frequency_start}"
                    )
            else:
                fas[i] = _fs
    fbs = fas + ffs
    # Return domain data
    return {
        "ns": ns,
        "tas": tas, "tbs": tbs, "dts": dts, "tts": tts,
        "fas": fas, "fbs": fbs, "dfs": dfs, "ffs": ffs,
    }


def _get_fft_domain_phase_correction(ns, grads, ad_ndim, fft_axes, scalar=0):
    """
    Gets a phase correction array for custom domain ranges.

    Returns phases excluding 2π (i.e. f_k*t_m term of exp(-2πi f_k t_m)).
    """
    phase_correction = scalar
    for i, (n, grad) in enumerate(zip(ns, grads)):
        dim = fft_axes[i]
        if grad != 0:
            # Phase gradients
            _corr = (np.arange(n) * grad)
            # Broadcasting
            _idx = tuple([slice(None)] + (ad_ndim - dim - 1) * [np.newaxis])
            _corr = _corr[_idx]
            # Correction
            phase_correction = phase_correction + _corr
    if np.isscalar(phase_correction) and np.isclose(phase_correction, 0):
        return None
    else:
        return phase_correction


def _get_fft_var_quantity(q):
    """
    Inverts a quantity.
    """
    ft_q = copy.deepcopy(q)
    # Convert name
    if isinstance(q.name, str):
        qname = q.name.strip(" \t")
        found_repl = False
        patterns = [
            [r"^inverse (.*?)", "{}"],
            [r"(.*)\s+position$", "{} spatial frequency"],
            [r"(.*)\s+spatial frequency$", "{} position"],
            [r"(.*)\s+time$", "{} frequency"],
            [r"(.*)\s+frequency$", "{} time"],
        ]
        for pat, repl in patterns:
            match = re.match(pat, qname, flags=re.IGNORECASE)
            if match:
                ft_q.name = repl.format(match.group(1))
                found_repl = True
                break
        if not found_repl:
            ft_q.name = f"inverse {qname}"
    # Convert symbol
    if isinstance(q.symbol, str):
        qsym = q.symbol.strip(" \t")
        found_repl = False
        patterns = [
            [r"^1\s*/\s*\(\s*(.*?)\s*\)$", "{}"],
            [r"^1\s*/\s*\[\s*(.*?)\s*\]$", "{}"],
            [r"^1\s*/\s*(.*)$", "{}"],
            [r"^\(\s*(.*?)\s*\)\^\{\s*-\s*1\s*\}$", "{}"],
            [r"^\(\s*(.*?)\s*\)\^\(\s*-\s*1\s*\)$", "{}"],
            [r"^(\w+)\^\{\s*-\s*1\s*\}$", "{}"],
        ]
        for pat, repl in patterns:
            match = re.match(pat, qsym, flags=re.IGNORECASE)
            if match:
                ft_q.symbol = repl.format(match.group(1))
                found_repl = True
                break
        if not found_repl:
            patterns = [
                [r"^\s*t\s*$", "f"],
                [r"^\s*f\s*$", "t"]
            ]
            for pat, repl in patterns:
                match = re.match(pat, qsym, flags=re.IGNORECASE)
                if match:
                    ft_q.symbol = repl
                    found_repl = True
                    break
        if not found_repl:
            match = re.match(r"^\s*(\w+)\s*$", qsym, flags=re.IGNORECASE)
            if match:
                ft_q.symbol = f"1 / {match.group(1)}"
            else:
                ft_q.symbol = f"1 / ({qsym})"
    # Convert unit
    if isinstance(q.unit, str):
        qunit = q.unit.strip(" \t")
        found_repl = False
        patterns = [
            [r"^1\s*/\s*\(\s*(.*?)\s*\)$", "{}"],
            [r"^1\s*/\s*\[\s*(.*?)\s*\]$", "{}"],
            [r"^1\s*/\s*(.*)$", "{}"],
            [r"^\(\s*(.*?)\s*\)\^\{\s*-\s*1\s*\}$", "{}"],
            [r"^\(\s*(.*?)\s*\)\^\(\s*-\s*1\s*\)$", "{}"],
            [r"^(\w+)\^\{\s*-\s*1\s*\}$", "{}"],
        ]
        for pat, repl in patterns:
            match = re.match(pat, qunit, flags=re.IGNORECASE)
            if match:
                ft_q.unit = repl.format(match.group(1))
                found_repl = True
                break
        if not found_repl:
            patterns = [
                [r"^\s*(a|p|n|u|µ|m||k|M|G|T|P)s\s*$", "{}Hz"],
                [r"^\s*(a|p|n|u|µ|m||k|M|G|T|P)(?:H|h)z\s*$", "{}s"]
            ]
            repl_map = {"a": "P", "p": "T", "n": "G", "u": "M", "µ": "M",
                        "m": "k", "": ""}
            for k, v in list(repl_map.items()):
                repl_map[v] = k
            for pat, repl in patterns:
                match = re.match(pat, qunit)
                if match:
                    ft_q.unit = repl.format(repl_map[match.group(1)])
                    found_repl = True
                    break
        if not found_repl:
            match = re.match(r"^\s*(\w+)\s*$", qunit, flags=re.IGNORECASE)
            if match:
                ft_q.unit = f"1 / {match.group(1)}"
            else:
                ft_q.unit = f"1 / ({qunit})"
    # Return converted quantity
    return ft_q
