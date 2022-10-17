import abc
import copy
import numpy as np
import scipy.optimize
import scipy.stats
try:
    from scipy.stats._distn_infrastructure import rv_continuous_frozen
except ImportError:
    # Backward compatibilty after scipy update (commit on 2022-03-17)
    from scipy.stats._distn_infrastructure import (
        rv_frozen as rv_continuous_frozen
    )

from libics.env import logging
from libics.core.data.arrays import ArrayData, SeriesData
from libics.core.util import misc
from libics.core.io import FileBase


###############################################################################
# Fitting models
###############################################################################


class ModelBase(abc.ABC, FileBase):

    """
    Base class for functional models.

    * Supports fitting.
    * For convenience, fitting can be directly applied by passing the data
      to the constructor.

    Parameters
    ----------
    *data : `Any`
        Data interpretable by :py:meth:`_split_fit_data`.
    const_p0 : `dict(str->float)`
        Not fitted parameters set to the given value.
    **kwargs
        Keyword arguments passed to :py:func:`scipy.optimize.curve_fit`.

    Examples
    --------
    Assumes a concrete model class (i.e. subclass of this base class).

    >>> class Model(ModelBase):
    ...     P_ALL = ["x0"]
    ...     P_DEFAULT = [1]
    ...     @staticmethod
    ...     def _func(var, x0):
    ...         return var + x0
    >>> model = Model()
    >>> model.pfit = ["x0"]
    >>> model.p0 = [1]
    >>> var_data, func_data = np.array([[0, 1, 2], [2.2, 3.0, 3.8]])
    >>> model.find_popt(var_data, func_data)
    True
    >>> model.popt
    array([2.])
    >>> model.pstd
    array([0.013333333])
    >>> model.x0
    2.
    >>> model.x0_std
    0.013333333
    >>> model["x0"]
    2.
    >>> model["x0_std"]
    0.013333333
    >>> model(var_data)
    array([2., 3., 4.])

    A direct fit is applied as:

    >>> model = Model(var_data, func_data)
    >>> model.psuccess
    True

    Notes
    -----
    For subclassing, implement the following attributes and methods:

    * `P_ALL` : `Iterable[str]`, e.g. `["x0", "y0", "z0"]`.
      Ordered (w.r.t model function) list of parameter names (will be used to
      access parameters). Class attribute.
    * `P_DEFAULT` : `Iterable[float]`.
      Default `p0` parameters.
    * `_func` : `Callable[var, *param]`.
      Model function. Static method.
    * `find_p0` : `Callable[*data]`.
      Automatized initial parameter finding routine. The argument `*data`
      should be interpretable by :py:meth:`_split_fit_data`. Optional method.
    """

    LOGGER = logging.get_logger("libics.math.models.ModelBase")
    SER_KEYS = FileBase.SER_KEYS | {
        "_pfit", "_p0", "_popt", "_pcov", "psuccess", "_var_rect"
    }

    # Ordered list of all parameter names
    P_ALL = NotImplemented
    P_DEFAULT = NotImplemented

    def __init__(self, *data, const_p0=None, **kwargs):
        # Parameter names to be fitted (map to _popt index)
        self._pfit = None
        # Initial fit parameters
        self._p0 = None
        # Optimized fit parameters
        self._popt = None
        # Covariance matrix of fit parameters
        self._pcov = None
        # Flag whether fit succeeded
        self.psuccess = None
        # Domain of data variables
        self._var_rect = None

        # Call fit functions if data is supplied
        if len(data) > 0:
            self.find_p0(*data)
            if const_p0 is not None:
                self.set_pfit(**const_p0)
            self.find_popt(*data, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.P_ALL is NotImplemented:
            raise NotImplementedError("Class attribute `P_ALL` is missing")
        if (
            cls.P_DEFAULT is NotImplemented
            or len(cls.P_DEFAULT) != len(cls.P_ALL)
        ):
            raise NotImplementedError("Class attribute `P_DEFAULT` is missing")

    @staticmethod
    @abc.abstractmethod
    def _func(var, *p):
        """
        Model function.

        Parameters
        ----------
        var : `np.ndarray`
            Variables.
        *p : `float`
            Parameters in the order defined by :py:attr:`pall`.
        """
        raise NotImplementedError

    # +++++++++++++++++++++++++++++++++++++++
    # Properties API
    # +++++++++++++++++++++++++++++++++++++++

    @property
    def pall(self):
        """dict(name->index) for all parameters"""
        return misc.make_dict(self.P_ALL, range(len(self.P_ALL)))

    @property
    def pfit(self):
        """dict(name->index) for fitted parameters"""
        # By default fit all parameters
        if self._pfit is None:
            return self.pall.copy()
        else:
            return self._pfit

    @pfit.setter
    def pfit(self, val):
        # Check arguments
        for v in val:
            if v not in self.P_ALL:
                raise ValueError(f"invalid parameter name {v}")
        # Set fit parameters
        if isinstance(val, dict):
            self._pfit = val
        else:
            self._pfit = misc.make_dict(val, range(len(val)))

    def set_pfit(self, *opt, const=None, **const_p0):
        """
        Set which parameters to fit.

        Parameters
        ----------
        *opt : `str`
            Names of fit parameters to be optimized.
            If any `opt` is given, `const` and `const_p0` will be ignored.
        const : `Iter[str]`
            Names of parameters not to be fitted.
        **const_p0 : `str->float`
            Same as `const`, `float` arguments are used to fix the
            constant parameters.
        """
        if len(opt) > 0:
            self.pfit = opt
        else:
            _pfit = self.pall.copy()
            if len(const_p0) > 0:
                self.set_p0(**const_p0)
                for k in const_p0:
                    if k in _pfit:
                        del _pfit[k]
            if const is not None:
                if isinstance(const, str):
                    const = [const]
                for k in const:
                    if k in _pfit:
                        del _pfit[k]
            self.pfit = list(_pfit)

    @property
    def p0(self):
        """All :py:attr:`p0`"""
        return (
            self._p0 if self.p0_is_set()
            else np.array(self.P_DEFAULT, dtype=float)
        )

    @p0.setter
    def p0(self, val):
        if isinstance(val, dict):
            _p0 = self.get_p0(as_dict=True)
            _p0.update(val)
            val = [_p0[k] for k in self.pall]
        self._p0 = np.array(val, dtype=float)

    def get_p0(self, as_dict=True):
        if as_dict:
            return misc.make_dict(self.pall, self.p0)
        else:
            return self.p0

    def set_p0(self, **p0):
        """
        Sets initial parameters.

        Parameters
        ----------
        **p0 : `str->float`
            Initial parameter name->value.
        """
        self.p0 = p0

    def p0_is_set(self):
        return self._p0 is not None

    @property
    def p0_for_fit(self):
        """Fitted :py:attr:`p0`"""
        p0 = copy.deepcopy(self.p0)
        p0_for_fit = len(self.pfit) * [None]
        for k, i in self.pfit.items():
            p0_for_fit[i] = p0[self.pall[k]]
        return np.array(p0_for_fit, dtype=float)

    @property
    def popt(self):
        """All :py:attr:`popt` (non-fitted ones use :py:attr:`p0`)"""
        popt = self.p0.copy()
        if self._popt is not None:
            for k, i in self.pfit.items():
                popt[self.pall[k]] = self._popt[i]
        return np.array(popt, dtype=float)

    def get_popt(self, as_dict=True, pall=True):
        popt = self.popt if pall else self._popt.copy()
        if as_dict:
            pnames = self.pall if pall else self.pfit
            return misc.make_dict(pnames, popt)
        else:
            return popt

    @property
    def popt_for_fit(self):
        """Fitted :py:attr:`popt`"""
        return self._popt

    @popt_for_fit.setter
    def popt_for_fit(self, val):
        self._popt = np.array(val, dtype=float)

    @property
    def pcov(self):
        return self._pcov

    @property
    def pcov_for_fit(self):
        return self._pcov

    @pcov_for_fit.setter
    def pcov_for_fit(self, val):
        self._pcov = np.array(val, dtype=float)

    @property
    def pstd(self):
        # TODO: Get pall with dummy std for p0 only
        # TODO: Same for pcov
        var = np.diag(self.pcov)
        if np.any(var < 0):
            return np.full_like(var, np.inf, dtype=float)
        else:
            return np.sqrt(var)

    @property
    def pstd_for_fit(self):
        return np.sqrt(np.diag(self.pcov_for_fit))

    def get_pstd(self, as_dict=True):
        if as_dict:
            return misc.make_dict(self.pfit, self.pstd)
        else:
            return self.pstd

    def __getattr__(self, name):
        try:
            if name[-4:] == "_std":
                # return self.pstd[self.pall[name]]  # TODO: fix property pstd
                return self.pstd[self.pfit[name[:-4]]]
            else:
                return self.popt[self.pall[name]]
        except KeyError:
            return super().__getattribute__(name)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key[-4:] == "_std":
                return self.pstd[self.pall[key[:-4]]]
            else:
                return self.popt[self.pall[key]]
        else:
            return self.popt[key]

    def __str__(self):
        return f"{self.__class__.__name__}: {self.get_popt(as_dict=True)}"

    def __repr__(self):
        return f"<{str(self)}>"

    # +++++++++++++++++++++++++++++++++++++++
    # Methods API
    # +++++++++++++++++++++++++++++++++++++++

    def find_p0(self):
        """
        Routine for initial fit parameter finding.

        Should return whether this succeeded.
        """
        raise NotImplementedError

    def find_popt(self, *data, **kwargs):
        """
        Fits the model function to the given data.

        Parameters
        ----------
        *data : `Any`
            Data interpretable by :py:meth:`_split_fit_data`.
        **kwargs
            Keyword arguments passed to :py:func:`scipy.optimize.curve_fit`.

        Returns
        -------
        psuccess : `bool`
            Whether fit succeeded.
        """
        # Define (reduced) fit function
        _p = np.copy(self.p0)
        _pall = self.pall
        _pfit = self.pfit
        _func = self._func

        def _fit_func(var, *p):
            for k_fit, i_fit in _pfit.items():
                _p[_pall[k_fit]] = p[i_fit]
            return _func(var, *_p).ravel()

        # Prepare fit data
        split_data = self.split_fit_data(*data)
        var_data, func_data, err_data = self._ravel_data(*split_data)
        if err_data is not None and "sigma" not in kwargs:
            if err_data.dtype != float:
                err_data = err_data.astype(float)
            kwargs.update({"sigma": err_data})
        if var_data.dtype != float:
            var_data = var_data.astype(float)
        if func_data.dtype != float:
            func_data = func_data.astype(float)
        p0 = np.copy(self.p0_for_fit)

        # Optimize parameters
        try:
            self.popt_for_fit, self.pcov_for_fit = scipy.optimize.curve_fit(
                _fit_func, var_data, func_data, p0=p0, **kwargs
            )
            self.psuccess = (
                not np.any(np.isnan(self.pcov_for_fit))
                and np.all(np.isfinite(self.pcov_for_fit))
            )
        except RuntimeError:
            self.psuccess = False
        return self.psuccess

    def find_chi2(self, *data):
        """
        Gets the chi squared statistic.

        This is the sum of the squared residuals.
        """
        split_data = self._split_fit_data(*data)
        var_data, func_data, err_data = self._ravel_data(*split_data)
        if err_data is None:
            raise ValueError("No error specified")
        diff_data = func_data - self.__call__(var_data).ravel()
        chi2 = np.sum(diff_data**2 / err_data**2)
        return chi2

    def find_chi2_red(self, *data):
        """
        Gets the reduced chi squared statistic.

        This is the sum of the squared residuals divided by the degrees of
        freedom (data points minus number of fit parameters).
        """
        chi2 = self.find_chi2(*data)
        chi2_red = chi2 / (data[0].size - len(self.pfit))
        return chi2_red

    def find_chi2_significance(self, *data):
        """
        Gets the chi squared confindence quantile for the fit.

        Example: For a 2-sigma-quality fit, `0.95` is returned.
        """
        chi2_red = self.find_chi2_red(*data)
        dof = data[0].size - len(self.pfit)
        if dof > 50:
            quantile = 1 - scipy.stats.normal.cdf(chi2_red, np.sqrt(2 / dof))
        else:
            quantile = 1 - scipy.stats.chi2.cdf(chi2_red * dof, dof)
        return quantile

    def test_hypothesis_chi2(self, *data, p_value=0.05):
        """
        Tests the whether the fit is valid.

        Parameters
        ----------
        p_value : `float`
            Critical confidence level ("alpha-value"),
            e.g. `0.05` for 2-sigma.
        """
        chi2_red = self.find_chi2_red(*data)
        dof = data[0].size - len(self.pfit)
        if dof > 50:
            crit = scipy.stats.normal.ppf(1 - p_value, np.sqrt(2 / dof))
        else:
            crit = scipy.stats.chi2.ppf(1 - p_value, dof) / dof
        return chi2_red <= crit

    def __call__(self, var, *args, **kwargs):
        """
        Calls the model function with current parameters.

        Parameters
        ----------
        var : `np.ndarray` or `ArrayData`
            Variables.
            If `ArrayData`, uses its `var_meshgrid` as variables
            and overwrites its data.
        *args : `Any`
            Model function positional arguments after the parameters.
        **kwargs : `Any`
            Parameters or model function keyword arguments.
            If a parameter is given, the respective parameter is overwritten
            for the call.

        Returns
        -------
        res : `np.ndarray` or `ArrayData`
            Model function return value.
            If applicable, is wrapped in an `ArrayData`.
        """
        # Parse arguments
        ad = None
        if isinstance(var, ArrayData):
            ad = var
            var = ad.get_var_meshgrid()
        # Get current parameters
        _popt = self.popt
        if _popt is None:
            p = self.p0
            self.LOGGER.info(f"Call using `p0` = {str(p)}")
        else:
            p = _popt
        # Overwrite parameters with external arguments
        for k in list(kwargs.keys()):
            if k in self.pall:
                p[self.pall[k]] = kwargs[k]
                del kwargs[k]
        # Set return value depending on arguments
        res = self._func(var, *p, *args, **kwargs)
        if ad is None:
            return res
        else:
            ad.data = res
            return ad

    def get_model_data(self, shape=50):
        """
        Gets a (continuos) model data array using fitted parameters.

        Generates a continuos meshgrid spanning the domain of the data points
        used for fitting. Calls the model function using the optimized
        fit parameters for this meshgrid.

        Parameters
        ----------
        shape : `Iter[int]` or `int`
            Shape of meshgrid. If `int`, uses the same size in all dimensions.

        Returns
        -------
        data_cont : `ArrayData(float)`
            Continuos model data.

        Raises
        ------
        ValueError
            If no prior fit has been performed.
        """
        # Parse parameters
        if self._var_rect is None:
            raise ValueError("No prior fit has been performed")
        if np.isscalar(shape):
            shape = np.full(len(self._var_rect), np.ceil(shape), dtype=int)
        else:
            shape = np.round(shape).astype(int)
        # Generate meshgrid
        ndim = len(shape)
        data_cont = ArrayData()
        for _low, _high in self._var_rect:
            data_cont.add_dim(low=_low, high=_high)
        data_cont.var_shape = tuple(shape)
        # Call model function
        if ndim == 1:
            data_cont.data = self(data_cont.get_points(0))
        else:
            data_cont.data = self(data_cont.get_var_meshgrid())
        return data_cont

    # +++++++++++++++++++++++++++++++++++++++
    # Helper functions
    # +++++++++++++++++++++++++++++++++++++++

    def split_fit_data(self, *data, func_dim=-1):
        var_data, func_data, err_data = self._split_fit_data(
            *data, func_dim=func_dim
        )
        self._var_rect = [[np.min(v), np.max(v)] for v in var_data]
        return var_data, func_data, err_data

    @staticmethod
    def _split_fit_data(*data, func_dim=-1):
        """
        Splits the given data into independent and dependent data, as required
        by the fit class.

        Parameters
        ----------
        *data : `ArrayData` or `SeriesData` or `np.ndarray`
            Data to be split.
            If two arrays are given, they are interpreted as
            `(var_data, func_data)`.
            If three arrays are given, they are interpreted as
            `(var_data, func_data, err_data)`.
        func_dim : `int`
            Only used for `SeriesData`.
            Dependent data dimension (index).

        Returns
        -------
        var_data : `np.ndarray`
            Independent data.
        func_data : `np.ndarray`
            Dependent data.
        func_err : `np.ndarray` or `None`
            Errors of dependent data (if available).
        """
        var_data, func_data, err_data = None, None, None
        # Single argument
        if len(data) == 1:
            data = data[0]
            if isinstance(data, SeriesData):
                func_data = data.data[func_dim]
                var_data = np.concatenate(
                    (data.data[0:func_dim], data.data[func_dim + 1:])
                )
            else:
                if isinstance(data, ArrayData):
                    func_data = data.data
                    var_data = data.get_var_meshgrid()
                else:
                    func_data = np.array(data)
                    var_data = np.indices(func_data.shape)
        elif len(data) in [2, 3]:
            if isinstance(data[0], ArrayData):
                var_data = data[0].data
            else:
                var_data = np.array(data[0])
            if var_data.ndim == 1:
                var_data = var_data[np.newaxis, ...]
            if isinstance(data[1], ArrayData):
                func_data = data[1].data
            else:
                func_data = np.array(data[1])
            if len(data) == 3:
                if isinstance(data[2], ArrayData):
                    err_data = data[2].data
                elif data[2] is not None:
                    err_data = np.array(data[2])
        else:
            raise NotImplementedError
        # Split data
        return var_data, func_data, err_data

    @staticmethod
    def _ravel_data(
        var_data, func_data=None, err_data=None, _check_shape=True
    ):
        """
        Serializes array-like (nD) data into series-like (1D) data.

        Parameters
        ----------
        var_data : `np.ndarray`
            Array-like independent data.
        func_data : `np.ndarray`
            Array-like dependent data.
        err_data : `np.ndarray`
            Array-like dependent errors.
        _check_shape : `bool`
            Flag whether to check `var_data` and `func_data`
            shape overlap.

        Returns
        -------
        var_data : `np.ndarray`
            Serialized independent data.
        func_data : `np.ndarray`
            Serialized dependent data.
        err_data : `np.ndarray`
            Serialized dependent errors.

        Raises
        ------
        ValueError
            If data dimensions are invalid.
        """
        var_data = np.array(var_data)
        var_data_ind = 1 if var_data.ndim > 1 else 0
        if func_data is not None:
            func_data = np.array(func_data)
            if (
                _check_shape and
                var_data.shape[var_data_ind:] != func_data.shape
            ):
                raise ValueError(
                    "invalid fit data dimensions: {:s}, {:s}"
                    .format(var_data.shape, func_data.shape)
                )
            func_data = func_data.ravel()
        if err_data is not None:
            err_data = np.array(err_data)
            if (
                _check_shape and
                var_data.shape[var_data_ind:] != err_data.shape
            ):
                raise ValueError(
                    "invalid fit data dimensions: {:s}, {:s}"
                    .format(var_data.shape, err_data.shape)
                )
            err_data = err_data.ravel()
        if var_data_ind == 0:
            var_data = var_data.ravel()
        else:
            var_data = var_data.reshape((var_data.shape[0], -1))
        return var_data, func_data, err_data


# +++++++++++++++++++++++++++++++++++++++++++++
# Model generators
# +++++++++++++++++++++++++++++++++++++++++++++


def ParamModelFromArray(
    ar, param_dims=None, interpolation="linear", extrapolation=True
):
    """
    Uses an interpolated array to define a parameterized model.

    Parameters
    ----------
    ar : `Array[float]`
        Array interpreted as functional values.
    param_dims : `Iter[int]`
        Array dimensions used as fitting parameters.
        If `None`, uses only first dimension as parameter.
    interpolation : `str`
        Interpolation mode: `"nearest", "linear"`.
    extrapolation : `bool` or `float`
        If `True`, extrapolates using the method given by `interpolation`.
        If `False`, raises an error upon extrapolation.
        If `float`, uses the given value as constant extrapolation value.

    Returns
    -------
    FitArrayData : `class`
        Generated model class (subclass of :py:class:`ModelBase`).
    """
    # Parse parameters
    if isinstance(ar, ArrayData):
        ad = ar.copy()
    else:
        ad = ArrayData(np.array(ar).copy())
        for dim in range(ad.ndim):
            ad.set_var_quantity(dim, name=chr(ord("a") + dim))
    # Distinguish variables and parameters
    if param_dims is None:
        param_dims = [0]
    if np.isscalar(param_dims):
        param_dims = [param_dims]
    var_dims = []
    for dim in range(ad.ndim):
        if dim not in param_dims:
            var_dims.append(dim)
    param_names = [ad.var_quantity[dim].name for dim in param_dims]
    if len(np.unique(param_names)) != len(param_names):
        raise ValueError("non-unique parameter names")

    # Create model class
    if len(var_dims) == 0:
        raise ValueError("no variable dimension set")
    elif len(param_dims) == 0:
        raise ValueError("no parameter dimension set")
    # Scalar variable
    elif len(var_dims) == 1:
        class FitArrayData(ModelBase):

            P_ALL = param_names
            P_DEFAULT = [ad.get_center(dim) for dim in param_dims]

            _data = ad
            _var_dims = var_dims
            _param_dims = param_dims

            @staticmethod
            def _func(var, *p):
                cls = FitArrayData
                params = []
                i_p = 0
                for dim in range(cls._data.ndim):
                    if dim in cls._var_dims:
                        params.append(var)
                    else:
                        params.append(np.full_like(var, p[i_p]))
                        i_p += 1
                params = np.array(params)
                return ad.interpolate(
                    params, mode=interpolation, extrapolation=extrapolation
                )

    # Tensorial variable
    else:
        class FitArrayData(ModelBase):

            P_ALL = param_names
            P_DEFAULT = [ad.get_center(dim) for dim in param_dims]

            _data = ad
            _var_dims = var_dims
            _param_dims = param_dims

            @staticmethod
            def _func(var, *p):
                cls = FitArrayData
                params = []
                i_var, i_p = 0, 0
                for dim in range(cls._data.ndim):
                    if dim in cls._var_dims:
                        params.append(var[i_var])
                        i_var += 1
                    else:
                        params.append(np.full_like(var[0], p[i_p]))
                        i_p += 1
                params = np.array(params)
                return ad.interpolate(
                    params, mode=interpolation, extrapolation=extrapolation
                )

    return FitArrayData


def ScaleModelFromArray(
    ar, scale_dims=None, offset_dims=None,
    interpolation="linear", extrapolation=True
):
    """
    Uses an interpolated array to define a variable-scaled model.

    Can be used to fit a scale and offset to the array variables.

    Parameters
    ----------
    ar : `Array[float]`
        Array interpreted as functional values.
    scale_dims, offset_dims : `Iter[int]`
        Array dimensions that should be scaled/offset.
        If `scale_dims is None`, scales all dimensions.
        If `offset_dims is None`, offsets none of the dimensions.
    interpolation : `str`
        Interpolation mode: `"nearest", "linear"`.
    extrapolation : `bool` or `float`
        If `True`, extrapolates using the method given by `interpolation`.
        If `False`, raises an error upon extrapolation.
        If `float`, uses the given value as constant extrapolation value.

    Returns
    -------
    FitArrayData : `class`
        Generated model class (subclass of :py:class:`ModelBase`).
    """
    # Parse parameters
    if isinstance(ar, ArrayData):
        ad = ar.copy()
    else:
        ad = ArrayData(np.array(ar).copy())
        for dim in range(ad.ndim):
            ad.set_var_quantity(dim, name=chr(ord("a") + dim))
    # Distinguish variables and parameters
    if scale_dims is None:
        scale_dims = np.arange(ad.ndim)
    elif np.isscalar(scale_dims):
        scale_dims = [scale_dims]
    if offset_dims is None:
        offset_dims = []
    elif np.isscalar(offset_dims):
        offset_dims = [offset_dims]
    param_names = (
        [f"{ad.var_quantity[dim].name}_scale" for dim in scale_dims]
        + [f"{ad.var_quantity[dim].name}_offset" for dim in offset_dims]
    )
    param_default = len(scale_dims) * [1] + len(offset_dims) * [0]
    if len(np.unique(param_names)) != len(param_names):
        raise ValueError("non-unique parameter names")

    # Create model class
    if len(param_names) == 0:
        raise ValueError("no parameter dimensions set")
    # Scalar variable
    elif ad.ndim == 1:
        class FitArrayData(ModelBase):

            P_ALL = param_names
            P_DEFAULT = param_default

            _data = ad
            _scale_dims = scale_dims
            _offset_dims = offset_dims

            @staticmethod
            def _func(var, *p):
                cls = FitArrayData
                _scale_ndim = len(cls._scale_dims)
                _offset_ndim = len(cls._offset_dims)
                if _scale_ndim == 1:
                    var = p[0] * var
                if _offset_ndim == 1:
                    var = p[_scale_ndim] + var
                return ad.interpolate(
                    var, mode=interpolation, extrapolation=extrapolation
                )

    # Tensorial variable
    else:
        class FitArrayData(ModelBase):

            P_ALL = param_names
            P_DEFAULT = param_default

            _data = ad
            _scale_dims = scale_dims
            _offset_dims = offset_dims

            @staticmethod
            def _func(var, *p):
                cls = FitArrayData
                params = []
                i_scale, i_offset = 0, 0
                for dim in range(cls._data.ndim):
                    _tmp = var[dim]
                    if dim in cls._scale_dims:
                        _tmp = p[i_scale] * _tmp
                        i_scale += 1
                    if dim in cls._offset_dims:
                        _tmp = _tmp + p[len(cls._scale_dims) + i_offset]
                        i_offset += 1
                    params.append(_tmp)
                params = np.array(params)
                return ad.interpolate(
                    params, mode=interpolation, extrapolation=extrapolation
                )

    return FitArrayData


###############################################################################
# Tensorial parameters
###############################################################################


class TensorModelBase(abc.ABC):

    LOGGER = logging.get_logger("libics.math.models.TensorModelBase")

    P_TENS = "ptens"
    P_VECT = NotImplemented
    P_SCAL = NotImplemented

    P_DEFAULT = NotImplemented

    def __init__(self, *data, **kwargs):
        self.LOGGER.warning("Class is untested!")
        # Initial fit parameters
        self._p0_tens = None
        self._p0_vect = None
        self._p0_scal = None
        # Optimized fit parameters
        self._popt_tens = None
        self._popt_vect = None
        self._popt_scal = None
        # Covariance matrix of fit parameters
        self._pcov = None
        # Flag whether fit succeeded
        self.psuccess = None

        # Call fit functions if data is supplied
        if len(data) > 0:
            self.find_p0(*data)
            self.find_popt(*data, **kwargs)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.P_VECT is NotImplemented:
            raise NotImplementedError("Class attribute `P_VECT` is missing")
        if cls.P_SCAL is NotImplemented:
            raise NotImplementedError("Class attribute `P_SCAL` is missing")
        if cls.P_DEFAULT is NotImplemented:
            raise NotImplementedError("Class attribute `P_DEFAULT` is missing")

    @staticmethod
    @abc.abstractmethod
    def _func(var, *p):
        """
        Model function with signature `func(var, tens, *vect, *scal)`.
        """
        raise NotImplementedError

    # +++++++++++++++++++++++++++++++++++++++
    # Properties API
    # +++++++++++++++++++++++++++++++++++++++

    @property
    def pall_vect(self):
        """dict(name->index) for vector parameters"""
        return misc.make_dict(self.P_VECT, range(len(self.P_VECT)))

    @property
    def pall_scal(self):
        """dict(name->index) for scalar parameters"""
        return misc.make_dict(self.P_SCAL, range(len(self.P_SCAL)))

    @property
    def p0(self):
        if self.p0_is_set():
            return self._p0_tens, self._p0_vect, self._p0_scal
        else:
            return (
                np.array([self.P_DEFAULT[0]]),
                np.array(self.P_DEFAULT[1:1+len(self.P_VECT)], dtype=float),
                np.array(self.P_DEFAULT[1+len(self.P_VECT):], dtype=float)
            )

    @p0.setter
    def p0(self, val):
        self._p0_tens = np.array(val[0]) if val[0] is not None else val[0]
        self._p0_vect = np.array(val[1]) if val[1] is not None else val[1]
        self._p0_scal = np.array(val[2]) if val[2] is not None else val[2]

    def p0_is_set(self):
        return (
            self._p0_tens is not None
            and (self._p0_vect is not None or len(self.P_VECT) == 0)
            and (self._p0_scal is not None or len(self.P_SCAL) == 0)
        )

    @property
    def popt(self):
        """All :py:attr:`popt` (non-fitted ones use :py:attr:`p0`)"""
        if self.popt_is_set():
            return self._popt_tens, self._popt_vect, self._popt_scal
        else:
            return self.p0

    def popt_is_set(self):
        return (
            self._popt_tens is not None
            and (self._popt_vect is not None or len(self.P_VECT) == 0)
            and (self._popt_scal is not None or len(self.P_SCAL) == 0)
        )

    def get_popt(self, as_dict=True):
        tens_popt, vect_popt, scal_popt = self.popt
        if as_dict:
            d = {self.P_TENS: tens_popt}

            d.update(misc.make_dict(self.P_VECT, vect_popt))
            d.update(misc.make_dict(self.P_SCAL, scal_popt))
            return d
        else:
            return self.popt

    @property
    def pcov(self):
        return self._pcov

    @property
    def pstd(self):
        var = np.diag(self.pcov)
        if np.any(var < 0):
            return np.full_like(var, np.inf, dtype=float)
        else:
            return np.sqrt(var)

    def __getattr__(self, name):
        get_std = False
        if name[-4:] == "_std":
            get_std = True
            name = name[:-4]
        try:
            if get_std:
                raise NotImplementedError(f"{name}_std not implemented")
            if name == self.P_TENS:
                return self.popt[0]
            elif name in self.P_VECT:
                return self.popt[1][self.pall_vect[name]]
            elif name in self.P_SCAL:
                return self.popt[2][self.pall_scal[name]]
            else:
                return super().__getattribute__(name)
        except KeyError:
            return super().__getattribute__(name)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key[-4:] == "_std":
                return self.pstd[self.pall[key[:-4]]]
            else:
                return self.popt[self.pall[key]]
        else:
            return self.popt[key]

    def _str(self, join_str="\n → "):
        d = self.get_popt(as_dict=True)
        ptens = d.pop(self.P_TENS)
        pvect = {k: d[k] for k in self.P_VECT}
        pscal = {k: d[k] for k in self.P_SCAL}
        s = [f"{self.P_TENS}: shape: {ptens.shape}, vmean: {ptens.mean()}, "
             f"vmin: {ptens.min()}, vmax: {ptens.max()}"]
        for k, v in pvect.items():
            s.append(f"{k}: {v}")
        if len(self.P_SCAL) > 0:
            s.append(f"{pscal}")
        return join_str.join(s)

    def __str__(self):
        s = " → " + self._str(join_str="\n → ")
        return f"{self.__class__.__name__}:\n{s}"

    def __repr__(self):
        s = self._str(join_str='\n')
        return f"<'{self.__class__.__name__}' at {hex(id(self))}>\n{s}"

    # +++++++++++++++++++++++++++++++++++++++
    # Methods API
    # +++++++++++++++++++++++++++++++++++++++

    def find_p0(self):
        """
        Routine for initial fit parameter finding.

        Should return whether this succeeded.
        """
        raise NotImplementedError

    def find_popt(self, *data, **kwargs):
        """
        Fits the model function to the given data.

        Parameters
        ----------
        *data : `Any`
            Data interpretable by :py:meth:`_split_fit_data`.
        **kwargs
            Keyword arguments passed to :py:func:`scipy.optimize.curve_fit`.

        Returns
        -------
        psuccess : `bool`
            Whether fit succeeded.
        """
        # Find data structure
        var_data, func_data, err_data = self._split_fit_data(*data)
        # var_ndim = len(var_data)
        var_shape = var_data.shape
        # func_shape = func_data.shape
        # func_size = np.prod(func_shape)
        tens_shape = func_data.shape[1:]
        tens_size = np.prod(tens_shape)
        vect_num = len(func_data)
        vect_shape = (len(self.P_VECT), vect_num)
        vect_size = np.prod(vect_shape)
        # scal_num = len(self.P_SCAL)
        # scal_shape = (scal_num,)
        # scal_size = scal_num

        # Define (reduced) fit function
        def _fit_func(var, *p):
            # Structure of variables:
            # 2D: [[*var_data[0]], [*var_data[1]], ...]
            var = var.reshape(var_shape)
            # Structure of parameters:
            # 1D: [*array, *vect0, *vect1, ..., scal0, scal1, ...]
            p = np.array(p)
            tens = p[0:tens_size].reshape(tens_shape)
            vect = p[tens_size:tens_size+vect_size].reshape(vect_shape)
            scal = p[tens_size+vect_size:]
            # Call fit function
            res = self._func(var, tens, *vect, *scal)
            return res.ravel()

        # Prepare fit data
        var_data, func_data, err_data = self._ravel_data(
            var_data, func_data, err_data
        )
        if err_data is not None and "sigma" not in kwargs:
            kwargs.update({"sigma": err_data})
        p0 = np.concatenate([_p0.ravel() for _p0 in self.p0])

        # Optimize parameters
        try:
            popt, pcov = scipy.optimize.curve_fit(
                _fit_func, var_data, func_data, p0=p0, **kwargs
            )
        except RuntimeError:
            return False
        self.psuccess = (
            not np.any(np.isnan(pcov))
            and np.all(np.isfinite(pcov))
        )

        # Reshape parameters
        popt = (
            popt[0:tens_size].reshape(tens_shape),
            popt[tens_size:tens_size+vect_size].reshape(vect_shape),
            popt[tens_size+vect_size:]
        )
        pcov = (
            pcov[0:tens_size, 0:tens_size].reshape(tens_shape + tens_shape),
            (pcov[tens_size:tens_size+vect_size, tens_size:tens_size+vect_size]
             .reshape(vect_shape + vect_shape)),
            pcov[tens_size+vect_size:, tens_size+vect_size:]
        )
        self.popt = popt
        self.pcov = pcov
        return self.psuccess

    def __call__(self, var, *args, **kwargs):
        """
        Calls the model function with current parameters.

        Parameters
        ----------
        var : `np.ndarray` or `ArrayData`
            Variables.
            If `ArrayData`, uses its `var_meshgrid` as variables
            and overwrites its data.
        *args : `Any`
            Model function positional arguments after the parameters.
        **kwargs : `Any`
            Parameters or model function keyword arguments.
            If a parameter is given, the respective parameter is overwritten
            for the call.

        Returns
        -------
        res : `np.ndarray` or `ArrayData`
            Model function return value.
            If applicable, is wrapped in an `ArrayData`.
        """
        # Parse arguments
        ad = None
        if isinstance(var, ArrayData):
            ad = var
            var = ad.get_var_meshgrid()
        # Get current parameters
        ptens, pvect, pscal = self.popt
        # Overwrite parameters with external arguments
        for k, v in kwargs.items():
            if k == self.P_TENS:
                ptens = v
            elif k in self.P_VECT:
                pvect[self.pall_vect[k]] = v
                del kwargs[k]
            elif k in self.P_SCAL:
                pscal[self.pall_scal[k]] = v
                del kwargs[k]
        # Set return value depending on arguments
        res = self._func(var, ptens, *pvect, *pscal, *args, **kwargs)
        if ad is None:
            return res
        else:
            ad.data = res
            return ad

    # +++++++++++++++++++++++++++++++++++++++
    # Helper functions
    # +++++++++++++++++++++++++++++++++++++++

    @staticmethod
    def _split_fit_data(*data, func_dim=-1):
        """
        Splits the given data into independent and dependent data, as required
        by the fit class.

        Parameters
        ----------
        *data : `ArrayData` or `SeriesData` or `np.ndarray`
            Data to be split.
            If two arrays are given, they are interpreted as
            `(var_data, func_data)`.
            If three arrays are given, they are interpreted as
            `(var_data, func_data, err_data)`.
        func_dim : `int`
            Only used for `SeriesData`.
            Dependent data dimension (index).

        Returns
        -------
        var_data : `np.ndarray`
            Independent data.
        func_data : `np.ndarray`
            Dependent data.
        func_err : `np.ndarray` or `None`
            Errors of dependent data (if available).
        """
        var_data, func_data, err_data = None, None, None
        # Single argument
        if len(data) == 1:
            data = data[0]
            if isinstance(data, SeriesData):
                func_data = data.data[func_dim]
                var_data = np.concatenate(
                    (data.data[0:func_dim], data.data[func_dim + 1:])
                )
            else:
                if isinstance(data, ArrayData):
                    func_data = data.data
                    var_data = data.get_var_meshgrid()
                else:
                    func_data = np.array(data)
                    var_data = np.indices(func_data.shape)
        elif len(data) in [2, 3]:
            if isinstance(data[0], ArrayData):
                var_data = data[0].data
            else:
                var_data = np.array(data[0])
            if var_data.ndim == 1:
                var_data = var_data[np.newaxis, ...]
            if isinstance(data[1], ArrayData):
                func_data = data[1].data
            else:
                func_data = np.array(data[1])
            if len(data) == 3:
                if isinstance(data[2], ArrayData):
                    err_data = data[2].data
                elif data[2] is not None:
                    err_data = np.array(data[2])
        else:
            raise NotImplementedError
        # Split data
        return var_data, func_data, err_data

    @staticmethod
    def _ravel_data(
        var_data, func_data=None, err_data=None, _check_shape=True
    ):
        """
        Serializes array-like (nD) data into series-like (1D) data.

        Parameters
        ----------
        var_data : `np.ndarray`
            Array-like independent data.
        func_data : `np.ndarray`
            Array-like dependent data.
        err_data : `np.ndarray`
            Array-like dependent errors.
        _check_shape : `bool`
            Flag whether to check `var_data` and `func_data`
            shape overlap.

        Returns
        -------
        var_data : `np.ndarray`
            Serialized independent data.
        func_data : `np.ndarray`
            Serialized dependent data.
        err_data : `np.ndarray`
            Serialized dependent errors.

        Raises
        ------
        ValueError
            If data dimensions are invalid.
        """
        var_data = np.array(var_data)
        var_data_ind = 1 if var_data.ndim > 1 else 0
        if func_data is not None:
            func_data = np.array(func_data)
            if (
                _check_shape and
                var_data.shape[var_data_ind:] != func_data.shape
            ):
                raise ValueError(
                    "invalid fit data dimensions: {:s}, {:s}"
                    .format(var_data.shape, func_data.shape)
                )
            func_data = func_data.ravel()
        if err_data is not None:
            err_data = np.array(err_data)
            if (
                _check_shape and
                var_data.shape[var_data_ind:] != err_data.shape
            ):
                raise ValueError(
                    "invalid fit data dimensions: {:s}, {:s}"
                    .format(var_data.shape, err_data.shape)
                )
            err_data = err_data.ravel()
        if var_data_ind == 0:
            var_data = var_data.ravel()
        else:
            var_data = var_data.reshape((var_data.shape[0], -1))
        return var_data, func_data, err_data


###############################################################################
# Random variable models
###############################################################################


class RvContinuousFrozen(rv_continuous_frozen):

    def __getattr__(self, name):
        return lambda *args, **kwargs: (
            getattr(self.dist, name)(*args, *self.args, **kwargs, **self.kwds)
        )


class RvContinuous(scipy.stats.rv_continuous):

    LOGGER = logging.get_logger("libics.tools.math.models.RvContinuous")

    def _ipdf(self, p, *args, branch="left", tol=None):
        p = np.array(p, dtype=float)
        xm = self._mode(*args)
        if branch == "left":
            opt_x0 = xm - 1
        elif branch == "right":
            opt_x0 = xm + 1
        else:
            raise ValueError("Invalid `branch` (should be 'left' or 'right')")
        mask = (
            (p > 0) & (p < self._amplitude(*args) * (1 + 1e-5))
        )
        p_masked = p[mask]
        x = np.full_like(p, np.nan)
        if len(p_masked) == 0:
            self.LOGGER.error("`_ipdf`: Invalid probability densities given")
            if x.ndim == 0:
                return float(x)
            else:
                return x

        # Find solutions to inverse PDF
        def root_func(x):
            return p_masked - self._pdf(x, *args)
        opt_x0 = np.full_like(p_masked, opt_x0)
        res = scipy.optimize.root(root_func, opt_x0, tol=tol)
        x[mask] = res.x
        # Check results
        if np.any(np.isnan(x)):
            self.LOGGER.error("`_ipdf`: Invalid probability densities given")
        elif not res.success:
            self.LOGGER.warning(f"`_ipdf` optimization: {res.message}")
        if x.ndim == 0:
            return float(x)
        else:
            return x

    def _mode(self, *args):
        res = scipy.optimize.minimize(-self._pdf, self._mean(*args), args=args)
        return res.x

    def _amplitude(self, *args):
        return self._pdf(self._mode(*args), *args)

    def ipdf(self, p, *args, branch="left", tol=None, **kwds):
        """
        Mode of the given RV, i.e., the maximum location of the PDF.

        Parameters
        ----------
        p : `Array`
            Probability density.
        arg1, arg2, arg3,... : `Array`
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : `Array`, optional
            location parameter (default=0)
        scale : `Array`, optional
            scale parameter (default=1)
        branch : `str`
            Which branch of the PDF to select.
            Options for unimodal distributions: `"left", "right"`.

        Returns
        -------
        ipdf : `Array`
            Inverse of the probability density function.
        """
        args, loc, scale = self._parse_args(*args, **kwds)
        try:
            ipdf = self._ipdf(p * scale, *args, branch=branch, tol=tol)
        except TypeError:
            ipdf = self._ipdf(p * scale, *args, branch=branch)
        return loc + scale * ipdf

    def mode(self, *args, **kwds):
        """
        Mode of the given RV, i.e., the maximum location of the PDF.

        Parameters
        ----------
        arg1, arg2, arg3,... : `Array`
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)
        loc : `Array`, optional
            location parameter (default=0)
        scale : `Array`, optional
            scale parameter (default=1)

        Returns
        -------
        mode : `float`
            Mode of probability density function.
        """
        args, loc, scale = self._parse_args(*args, **kwds)
        return loc + scale * self._mode(*args)

    def amplitude(self, *args, **kwds):
        """
        Amplitude of the given RV, i.e., the maximum value of the PDF.

        Parameters
        ----------
        arg1, arg2, arg3,... : `Array`
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information)

        Returns
        -------
        amplitude : `float`
            Amplitude of probability density function.
        """
        args, loc, scale = self._parse_args(*args, **kwds)
        return self._amplitude(*args) / scale

    def freeze(self, *args, **kwds):
        return RvContinuousFrozen(self, *args, **kwds)

    def variance(self, *args, **kwargs):
        return self.var(*args, **kwargs)

    def skewness(self, *args, **kwargs):
        return float(self.stats(*args, moments="s", **kwargs))

    def kurtosis(self, *args, **kwargs):
        return float(self.stats(*args, moments="k", **kwargs))

    def separation_loc(
        self, distr_l=None, distr_r=None, amp_l=1, amp_r=1,
        args_l=tuple(), args_r=tuple(), kwargs_l={}, kwargs_r={}
    ):
        """
        Gets the position of best separation between two distributions.

        Parameters
        ----------
        distr_l, distr_r : `RvContinuous` or `RvContinuousFrozen` or `None`
            (Left, right) distribution. If `None`, uses the current object.
            If `RvContinuousFrozen`, overwrites `arg` and `kwarg`.
        amp_l, amp_r : `float`
            (Relative) amplitudes of the distributions.
            Used if one distribution contains "more" probability than another.
        args_l, args_r : `tuple(float)`
            Distribution arguments (left, right).
        kwargs_l, kwargs_r : `dict(str->Any)`
            Distribution keyword arguments (left, right).

        Returns
        -------
        xs : `float`
            Separation position.
        ql, qr : `float`
            Quantile of the (left, right) distribution on the "wrong" side.
        """
        # Parse parameters
        if distr_l is None:
            distr_l = self
        if distr_r is None:
            distr_r = self
        if np.isscalar(args_l):
            args_l = (args_l,)
        if np.isscalar(args_r):
            args_r = (args_r,)
        if not isinstance(distr_l, rv_continuous_frozen):
            distr_l = distr_l(*args_l, **kwargs_l)
        if not isinstance(distr_r, rv_continuous_frozen):
            distr_r = distr_r(*args_r, **kwargs_r)
        # Minimize overlap
        return self.separation_loc_multi(
            [distr_l], [distr_r], amp_l=amp_l, amp_r=amp_r
        )

    @staticmethod
    def separation_loc_multi(distrs_l, distrs_r, amp_l=1, amp_r=1):
        """
        Gets the position of best separation between two distributions.

        Parameters
        ----------
        distr_l, distr_r : `RvContinuous` or `RvContinuousFrozen` or `None`
            (Left, right) distribution. If `None`, uses the current object.
            If `RvContinuousFrozen`, overwrites `arg` and `kwarg`.
        amp_l, amp_r : `float`
            (Relative) amplitudes of the distributions.
            Used if one distribution contains "more" probability than another.
        args_l, args_r : `tuple(float)`
            Distribution arguments (left, right).
        kwargs_l, kwargs_r : `dict(str->Any)`
            Distribution keyword arguments (left, right).

        Returns
        -------
        xs : `float`
            Separation position.
        ql, qr : `float`
            Quantile of the (left, right) distribution on the "wrong" side.
        """
        # Parse parameters
        try:
            distrs_l = list(distrs_l)
        except TypeError:
            distrs_l = [distrs_l]
        try:
            distrs_r = list(distrs_r)
        except TypeError:
            distrs_r = [distrs_r]
        for distr in misc.flatten_nested_list([distrs_l, distrs_r]):
            if not isinstance(distr, rv_continuous_frozen):
                raise ValueError("Distributions must be frozen")
        if np.isscalar(amp_l):
            amp_l = np.full(len(distrs_l), amp_l, dtype=float)
        if np.isscalar(amp_r):
            amp_r = np.full(len(distrs_r), amp_r, dtype=float)
        rel_amp_l, rel_amp_r = amp_l / np.sum(amp_l), amp_r / np.sum(amp_r)
        # Check if left/right distributions need to be swapped
        medians_l = [distr.median() for distr in distrs_l]
        medians_r = [distr.median() for distr in distrs_r]
        median_l = np.sum(np.array(medians_l) * rel_amp_l)
        median_r = np.sum(np.array(medians_r) * rel_amp_r)
        if median_l > median_r:
            RvContinuous.LOGGER.warning(
                "separation_loc_multi: flipped distribution order"
            )
            distrs_l, distrs_r = distrs_r, distrs_l
            median_l, median_r = median_r, median_l

        # Minimal overlap condition: equal weights across separation line `xs`
        def root_func(x):
            q_l = np.sum([
                amp * distr.sf(x) for amp, distr in zip(amp_l, distrs_l)
            ])
            q_r = np.sum([
                amp * distr.cdf(x) for amp, distr in zip(amp_r, distrs_r)
            ])
            return q_l - q_r
        # Minimize overlap
        bracket = [median_l, median_r]
        res = scipy.optimize.root_scalar(root_func, bracket=bracket)
        xs = res.root
        if not res.converged:
            RvContinuous.LOGGER.warning(
                "`separation_loc_multi` did not converge"
            )
        ql = np.sum([
            p * distr.sf(xs) for p, distr in zip(rel_amp_l, distrs_l)
        ])
        qr = np.sum([
            p * distr.cdf(xs) for p, distr in zip(rel_amp_r, distrs_r)
        ])
        return xs, ql, qr
