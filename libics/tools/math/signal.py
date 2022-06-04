import itertools
import numpy as np
from scipy import ndimage, optimize, signal

from libics.env import logging
from libics.core.data.arrays import ArrayData
from libics.core.util import misc
from libics.tools.math.models import ModelBase
from libics.tools.math.peaked import (
    FitGaussian1d, FitLorentzian1dAbs,
    FitSkewGaussian1d, SkewNormal1dDistribution
)
from libics.tools import plot


###############################################################################
# Histogram
###############################################################################


def find_histogram(data, **kwargs):
    """
    Finds a 1D histogram.

    Parameters
    ----------
    data : `Array[float]`
        Array on which to calculate histogram.
    **kwargs
        Keyword arguments passed to `np.histogram`.

    Returns
    -------
    hist : `ArrayData(1, float or int)`
        (Flattened) histogram.
    """
    ar = np.array(data).ravel()
    _h, _e = np.histogram(ar, **kwargs)
    _c = (_e[1:] + _e[:-1]) / 2
    hist = ArrayData(_h)
    hist.set_dim(0, points=_c)
    return hist


###############################################################################
# Peak detection
###############################################################################


LOGGER_PEAKS = logging.get_logger("libics.tools.math.signal.peaks")


def find_peaks_1d(
    *data, npeaks=None, rel_prominence=0.55, check_npeaks=True,
    base_prominence_ratio=None,
    edge_peaks=False, fit_algorithm="gaussian", ret_vals=None, _DEBUG=False
):
    """
    Finds the peaks of a 1D array by peak prominence.

    Parameters
    ----------
    *data : `Array[1, float]` or `(Array[1, float], Array[1, float])`
        1D data to find peaks in.
        Data may be provided as `ArrayData`, `data` or `x, y`.
    npeaks : `int` or `None`
        Number of peaks to be found.
        If `None`, determines `npeaks` from `rel_prominence`.
    rel_prominence : `float`
        Minimum relative prominence of peak w.r.t. mean of other found peaks.
    base_prominence_ratio : `float` or `None`
        If `None`: the size of the base is unrestricted.
        If `float`: the base of a peak may not extend into the base of
        secondary peaks if these peaks have a relative prominence that is
        higher than `base_prominence_ratio`.
    edge_peaks : `bool`
        Whether to allow the array edge to be considered as peak.
    check_npeaks : `bool` or `str`
        If `npeaks` is given, checks that `npeaks` is smaller or equal to
        the number of peaks found by `rel_prominence`.
        Performs the following actions if this condition is not fulfilled:
        If `error`, raises `RuntimeError`.
        If `warning` or `True`, prints a warning.
        If `False`, does not perform the check.
    fit_algorithm : `str` or `type(ModelBase)`
        Which algorithm to use for peak fitting among:
        `"mean", "gaussian", "lorentzian"`.
        Alternatively, a model class can be given. It must be subclassed
        from `libics.tools.models.ModelBase`; its center parameter must be
        called `x0`, its width `wx`.
    ret_vals : `Iter[str]`
        Sets which values to return. Available options:
        `"width", "fit"`.

    Returns
    -------
    result : `dict(str->list(float))`
        Dictionary containing the following items:
    center, center_err
        Peak positions and uncertainties.
    width
        Peak width.
    fit
        Fit object.
    """
    # Parse parameters
    if isinstance(ret_vals, str):
        ret_vals = [ret_vals]
    ret_vals = set(misc.assume_iter(ret_vals))
    result = {"center": [], "center_err": []}
    if "width" in ret_vals:
        result["width"] = []
    if "fit" in ret_vals:
        if fit_algorithm == "mean":
            raise ValueError("Cannot return `fit` for algorithm `mean`.")
        result["fit"] = []
    # Find peak slices
    peaks = find_peaks_1d_prominence(
        *data, npeaks=npeaks, rel_prominence=rel_prominence,
        base_prominence_ratio=base_prominence_ratio,
        edge_peaks=edge_peaks, check_npeaks=check_npeaks
    )
    if _DEBUG:
        plot.plot(data, color="black")
    # Peak by mean
    if fit_algorithm == "mean":
        _prob_min = np.mean([x.min() for x in peaks["data"]])
        for x in peaks["data"]:
            _prob = np.array(x) - _prob_min
            _prob = _prob / _prob.sum()
            _mean = (_prob * x.get_points(0)).sum()
            _std = np.sqrt((_prob * (x.get_points(0) - _mean)**2).sum())
            result["center"].append(_mean)
            result["center_err"].append(_std / np.sqrt(len(x)))
            if "width" in result:
                result["width"].append(_std)
        if _DEBUG:
            for i, _ad in enumerate(peaks["data"]):
                label = f"{i:d}: {peaks['prominence'][i]:.3f}"
                plot.plot(_ad, color=f"C{i:d}", label=label)
    # Peak by fit
    else:
        # Find fit class
        if (
            not isinstance(fit_algorithm, str)
            and issubclass(fit_algorithm, ModelBase)
        ):
            fit_class = fit_algorithm
        elif fit_algorithm == "gaussian":
            fit_class = FitGaussian1d
        elif fit_algorithm == "lorentzian":
            fit_class = FitLorentzian1dAbs
        else:
            raise ValueError(f"Invalid fit_algorithm ({str(fit_algorithm)})")
        # Perform fit
        fits = []
        for _ad in peaks["data"]:
            _fit = fit_class()
            _fit.find_p0(_ad)
            try:
                _fit.find_popt(_ad)
            except TypeError:
                _fit.find_p0(_ad)
                _fit.set_pfit(const=["c"])
                _fit.find_popt(_ad)
            if _fit.psuccess:
                fits.append(_fit)
            else:
                fits.append(None)
        if _DEBUG:
            for i, _ad in enumerate(peaks["data"]):
                _fit = fits[i]
                if _fit is not None:
                    _xcont = _ad.get_points(0)
                    _xcont = np.linspace(_xcont.min(), _xcont.max(), num=256)
                    label = f"{i:d}: {peaks['prominence'][i]:.3f}"
                    plot.plot(
                        _xcont, _fit(_xcont), color=f"C{i:d}", label=label
                    )
        # Get peaks from fits
        for _fit in fits:
            if _fit is not None:
                result["center"].append(_fit.x0)
                result["center_err"].append(_fit.x0_std)
                if "width" in result:
                    result["width"].append(_fit.wx)
                if "fit" in result:
                    result["fit"].append(_fit)
    if _DEBUG:
        ylim = plot.ylim()
        for i, (_c, _e) in enumerate(
            zip(result["center"], result["center_err"])
        ):
            color = f"C{i:d}"
            plot.axvline(_c, color=color)
            plot.fill_between([_c-_e, _c+_e], *ylim, color=color, alpha=0.2)
        plot.ylim(*ylim)
    return result


def find_peak_1d(*data):
    """
    Single-peak wrapper for :py:func:`find_peaks_1d`.

    Parameters
    ----------
    *data : `Array[1, float]` or `(Array[1, float], Array[1, float])`
        1D data to find peaks in.
        Data may be provided as `ArrayData`, `data` or `x, y`.

    Returns
    -------
    center, center_err : `float`
        Peak position and uncertainty.
    """
    result = find_peaks_1d(*data, npeaks=1, check_npeaks=False)
    return result["center"][0], result["center_err"][0]


def find_peaks_1d_prominence(
    *data, npeaks=None, rel_prominence=0.55,
    base_prominence_ratio=None, edge_peaks=False, check_npeaks="warning"
):
    """
    Finds positive peaks in 1D data by peak prominence.

    Parameters
    ----------
    *data : `Array[1, float]` or `(Array[1, float], Array[1, float])`
        1D data to find peaks in.
        Data may be provided as `ArrayData`, `data` or `x, y`.
    npeaks : `int` or `None`
        Number of peaks to be found.
        If `None`, determines `npeaks` from `rel_prominence`.
    rel_prominence : `float`
        Minimum relative prominence of peak w.r.t. mean of other found peaks.
    base_prominence_ratio : `float` or `None`
        If `None`: the size of the base is unrestricted.
        If `float`: the base of a peak may not extend into the base of
        secondary peaks if these peaks have a relative prominence that is
        higher than `base_prominence_ratio`.
    edge_peaks : `bool`
        Whether to allow the array edge to be considered as peak.
    check_npeaks : `bool` or `str`
        If `npeaks` is given, checks that `npeaks` is smaller or equal to
        the number of peaks found by `rel_prominence`.
        Performs the following actions if this condition is not fulfilled:
        If `error`, raises `RuntimeError`.
        If `warning` or `True`, prints a warning.
        If `False`, does not perform the check.

    Returns
    -------
    ret : `dict(str->Array[1, Any])`
        Dictionary containing the following items:
    position : `np.ndarray(1, float)`
        Raw position of each peak.
    data : `list(ArrayData)`
        List of sliced data containing the data of each peak.
    prominence : `np.ndarray(1, float)`
        Prominence of each peak.

    Note
    ----
    The number of returned peaks may vary.
    """
    # Parse data
    if len(data) == 2:
        ad = ArrayData(data[1])
        ad.set_dim(0, points=data[0])
    else:
        ad = ArrayData(data[0])
    ar = ad.data.copy()
    # Find raw peaks
    if edge_peaks and len(ar) > 1:
        edge_peak_left, edge_peak_right = ar[0] > ar[1], ar[-1] > ar[-2]
        if edge_peak_left:
            if edge_peak_right:
                ar = np.concatenate([[-np.inf], ar, [-np.inf]])
            else:
                ar = np.concatenate([[-np.inf], ar])
        else:
            ar = np.concatenate([ar, [-np.inf]])
    _peaks, _props = signal.find_peaks(ar, prominence=0)
    # Sort peaks by prominence
    _order = np.flip(np.argsort(_props["prominences"]))
    peaks, prominences, = _peaks[_order], _props["prominences"][_order]
    lb, rb = _props["left_bases"][_order], _props["right_bases"][_order]
    # Filter for requested peaks
    if npeaks is None or check_npeaks is not False:
        _npeaks = 1
        for i in range(_npeaks, len(peaks)):
            if prominences[i] >= rel_prominence * np.mean(prominences[:i]):
                _npeaks += 1
            else:
                break
        if npeaks is None:
            npeaks = _npeaks
        else:
            if _npeaks < npeaks:
                _msg = (
                    f"Expected `npeaks`={npeaks:d}, "
                    f"but found {_npeaks:d} peaks for "
                    f"`rel_prominence`={rel_prominence:.2f}"
                )
                if check_npeaks == "error":
                    raise RuntimeError(_msg)
                else:
                    LOGGER_PEAKS.warning(_msg)
    peaks = peaks[:npeaks]
    lb = lb[:npeaks]
    rb = rb[:npeaks]
    prominences = np.array(prominences[:npeaks])
    if edge_peaks:
        if edge_peak_left:
            peaks = [_peak - 1 for _peak in peaks]
            lb = [max(0, _b - 1) for _b in lb]
            rb = [_b - 1 for _b in rb]
            ar = ar[1:]
        if edge_peak_right:
            ar = ar[:-1]
        prominences[np.isinf(prominences)] = np.max(ar) - np.min(ar)
    # Check base overlap
    if base_prominence_ratio is not None:
        for i, _prom_primary in enumerate(prominences):
            _idxs_secondary = np.arange(i + 1, i + 1 + np.count_nonzero(
                prominences[i+1:] >= base_prominence_ratio * _prom_primary
            ))
            _peak_primary = peaks[i]
            for _idx_secondary in _idxs_secondary:
                if peaks[_idx_secondary] > _peak_primary:
                    if rb[i] > lb[_idx_secondary]:
                        rb[i] = lb[_idx_secondary]
                elif peaks[_idx_secondary] < _peak_primary:
                    if lb[i] < rb[_idx_secondary]:
                        lb[i] = rb[_idx_secondary]
        # Remove zero-length peaks
        mask_keep = np.array(lb) < np.array(rb)
        peaks = itertools.compress(peaks, mask_keep)
        lb, rb = np.array(lb)[mask_keep], np.array(rb)[mask_keep]
        prominences = np.array(prominences)[mask_keep]
    # Package data
    return {
        "position": np.array([ad.get_points(0)[idx] for idx in peaks]),
        "data": [ad[lb:rb+1] for lb, rb in zip(lb, rb)],
        "prominence": np.array(prominences)
    }


class PeakInfo:

    """
    Results class for peak analysis.

    Attributes
    ----------
    data : `ArrayData(1, float)`
        Raw peak data.
    fit : `FitSkewGaussian1d`
        Fit model object representing best skew Gaussian fit to the raw data.
    center : `float`
        (Fitted) maximum position of peak.
    width : `float`
        (Fitted) standard deviation of peak.
    base : `(float, float)`
        (Left, right) base position corresponding to the estimated
        range of the fit validity.
    subpeak : `None` or `PeakInfo`
        If `None`, no substructure in the peak is estimated.
        If `PeakInfo`, the :py:attr:`fit` may be insufficient to describe
        the peak, hence another subpeak is fitted (using only data outside
        of the :py:attr:`base`).
    """

    def __init__(
        self, data=None, fit=None, center=None, width=None, base=None,
        subpeak=None
    ):
        self.data = data
        self.fit = fit
        self.center = center
        self.width = width
        self.base = base
        self.subpeak = subpeak

    def iter_subpeaks(self):
        """
        Recursively iterates all subpeaks.

        Yields
        ------
        subpeak : `PeakInfo`
            Subpeak.
        """
        _current_peak = self
        while _current_peak.subpeak:
            _current_peak = _current_peak.subpeak
            yield _current_peak

    def iter_peaks(self):
        """
        Recursively iterates all peaks (both top-level peak and subpeaks).

        Yields
        ------
        peak : `PeakInfo`
            Peak.
        """
        yield(self)
        for subpeak in self.iter_subpeaks():
            yield subpeak

    @property
    def nsubpeaks(self):
        """Number of subpeaks."""
        return len(list(self.iter_subpeaks()))

    def get_model_data(self):
        """
        Gets the fitted peak data.

        Returns
        -------
        ad : `ArrayData(1, float)`
            Fitted peak data evaluated at the positions of the raw data.
        """
        x = self.data.get_points(0)
        y = np.zeros_like(x, dtype=float)
        for _peak in self.iter_peaks():
            y += _peak.fit(x)
        ad = ArrayData(y)
        ad.set_dim(0, points=x)
        return ad

    def __str__(self):
        centers, widths = [], []
        for subpeak in self.iter_peaks():
            centers.append(subpeak.center)
            widths.append(subpeak.width)
        centers, widths = np.array(centers), np.array(widths)
        nsubpeaks = len(centers) - 1
        if nsubpeaks == 0:
            centers, widths = centers[0], widths[0]
        _d = {"nsubpeaks": nsubpeaks, "center": centers, "width": widths}
        return f"{self.__class__.__name__}: {str(_d)}"

    def __repr__(self):
        return f"<{str(self)}>"

    @classmethod
    def minimal_overlap(cls, peak_info_left, peak_info_right):
        """
        Finds the position where the overlap between peaks is minimal.

        This position is calculated by making the absolute mass on the
        "wrong" side of each peak equal.
        Takes the total mass of each peak into account

        Parameters
        ----------
        peak_info_left, peak_info_right : `PeakInfo`
            Left and right peaks.

        Returns
        -------
        xs : `float`
            Peak separation position.
        overlapl, overlapr : `float`
            Probability for the (left, right) peak to be on the wrong side
            of the separation line (i.e., the separation error probability).
        """
        distr = SkewNormal1dDistribution
        # Separate peak infos
        pil = [_pi for _pi in peak_info_left.iter_peaks()]
        pir = [_pi for _pi in peak_info_right.iter_peaks()]
        # Get distribution parameters
        distrpl = []
        for _pi in pil:
            _d = _pi.fit.get_skew_normal_1d_distribution_params()
            distrpl.append((_d["x0"], _d["wx"], _d["alpha"]))
        distrpr = []
        for _pi in pir:
            _d = _pi.fit.get_skew_normal_1d_distribution_params()
            distrpr.append((_d["x0"], _d["wx"], _d["alpha"]))
        # Get distribution properties
        ampl = np.array([
            _pi.fit.a / distr.amplitude(*_dp[1:])
            for _pi, _dp in zip(pil, distrpl)
        ])
        ampr = np.array([
            _pi.fit.a / distr.amplitude(*_dp[1:])
            for _pi, _dp in zip(pir, distrpr)
        ])
        centerl = [_pi.center for _pi in pil]
        centerr = [_pi.center for _pi in pir]

        # Minimal overlap condition: equal weights across separation line `xs`
        def root_func(x):
            qls = [_a * (1 - distr.cdf(x, *_dp))
                   for _a, _dp in zip(ampl, distrpl)]
            qrs = [_a * distr.cdf(x, *_dp)
                   for _a, _dp in zip(ampr, distrpr)]
            ql, qr = np.sum(qls), np.sum(qrs)
            return ql - qr
        # Minimize overlap
        bracket = [np.min(centerl), np.max(centerr)]
        res = optimize.root_scalar(root_func, bracket=bracket)
        xs = res.root
        if not res.converged:
            cls.LOGGER.warning("`minimal_overlap` did not converge")
        # Estimate overlap probabilities
        overlapl = np.sum(
            ampl * np.array([1 - distr.cdf(xs, *_dp) for _dp in distrpl])
        ) / np.sum(ampl)
        overlapr = np.sum(
            ampr * np.array([distr.cdf(xs, *_dp) for _dp in distrpr])
        ) / np.sum(ampr)
        return xs, overlapl, overlapr


def analyze_single_peak(
    peak_ad, max_subpeaks=1, x0=None, c=0,
    max_width_std=1e-3, min_subpeak_len_abs=5, min_subpeak_len_rel=0.2,
    maxfev=10000, _is_recursion=False
):
    """
    Fits a peak with a skew Gaussian.

    Recursively fits subpeaks to residuals.

    Parameters
    ----------
    peak_ad : `Array[1, float]`
        Raw peak data.
    max_subpeaks : `int`
        Maximal number of subpeaks to be fitted.
    x0 : `float` or `None`
        If `float`, fixes the peak maximum position, thereby
        removing one fit degree of freedom. Only applied to top-level peak.
    c : `float`
        Fixed peak background level. Only applied to top-level peak.
    max_width_std : `float`
        Maximum fit uncertainty for the width to not continue to search
        for subpeaks.
    min_subpeak_len_abs : `int`
        Minimum absolute number of data points to attempt subpeak fit.
    min_subpeak_len_rel : `float`
        Minimum relative number of points outside the peak base
        to attempt subpeak fit.
    maxfev : `int`
        Maximum number of function evaluations for the fit.

    Returns
    -------
    peak_info : `PeakInfo` or `None`
        If a peak could be fitted, returns a peak information object.
        If not, returns `None`.
    """
    peak_ad = ArrayData(peak_ad)
    # Perform initial fit
    _fit = FitSkewGaussian1d()
    _fit.find_p0(peak_ad)
    const_p0 = {"c": c}
    if x0 is not None:
        const_p0["x0"] = x0
    _fit.set_pfit(**const_p0)
    _fit.find_popt(peak_ad, maxfev=maxfev)
    # Check if peak fit can be considered successful
    if (
        not _fit.psuccess
        or _fit.a < 0
        or (_is_recursion and "a" in _fit.pfit and _fit.a_std > _fit.a)
    ):
        return None
    else:
        # Set default peak base
        base_val = [peak_ad.get_low(0), peak_ad.get_high(0)]
        # Get peak fit residuals
        peak_ad_subtr = peak_ad - _fit(peak_ad.get_points(0))
        # Smoothen peaks for distribution parameter estimation
        _subtr_filtered = ndimage.gaussian_filter(peak_ad_subtr, 1)
        _subtr_filtered = ArrayData(_subtr_filtered)
        _subtr_filtered.set_dim(0, points=peak_ad_subtr.get_points(0))
        # Get distribution parameters (for statistical estimates)
        distr = SkewNormal1dDistribution
        distr_p = _fit.get_skew_normal_1d_distribution_params()
        _amp = _fit.a / distr.amplitude(distr_p["wx"], distr_p["alpha"])
        _p = distr_p["x0"], distr_p["wx"], distr_p["alpha"]
        # Find peak base by comparing to maximum of residuals
        distr_subtr_max = np.max(_subtr_filtered) / _amp
        try:
            base_val = [
                distr.ipdf(distr_subtr_max, *_p, branch=branch)
                for branch in ["left", "right"]
            ]
            check_subpeak = True
        except ValueError:
            check_subpeak = False
        if check_subpeak is False:
            subpeak = None
        else:
            # Check further whether subpeak fit is necessary
            if (
                max_subpeaks <= 0 or
                ("wx" in _fit.pfit and _fit.wx_std < max_width_std * _fit.wx)
            ):
                subpeak = None
            else:
                # Remove residuals data within peak base
                base_idx = [peak_ad_subtr.cv_quantity_to_index(val, 0)
                            for val in base_val]
                _d = peak_ad_subtr.data
                _d = np.concatenate([_d[:base_idx[0]+1], _d[base_idx[1]:]])
                _x = peak_ad_subtr.get_points(0)
                _x = np.concatenate([_x[:base_idx[0]+1], _x[base_idx[1]:]])
                # Check whether sufficient points are left
                if len(_d) < max(
                    min_subpeak_len_abs, len(peak_ad_subtr)*min_subpeak_len_rel
                ):
                    subpeak = None
                else:
                    # Recursively fit subpeak
                    peak_ad_subtr = ArrayData(_d)
                    peak_ad_subtr.set_dim(0, points=_x)
                    subpeak = analyze_single_peak(
                        peak_ad_subtr, max_subpeaks=max_subpeaks-1,
                        max_width_std=max_width_std,
                        min_subpeak_len_abs=min_subpeak_len_abs,
                        min_subpeak_len_rel=min_subpeak_len_rel,
                        maxfev=maxfev,
                        _is_recursion=True
                    )
    # Return result
    peak_info = PeakInfo(
        data=peak_ad,
        fit=_fit,
        center=_fit.x0,
        width=_fit.wx,
        base=np.array(base_val, dtype=float),
        subpeak=subpeak
    )
    return peak_info


###############################################################################
# Correlation
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
