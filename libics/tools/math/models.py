import abc
import copy
import numpy as np
import scipy.optimize

from libics.env import logging
from libics.core.data.arrays import ArrayData, SeriesData
from libics.core.util import misc


###############################################################################


class ModelBase(abc.ABC):

    """
    Base class for functional models.

    Supports fitting.

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

    # Ordered list of all parameter names
    P_ALL = NotImplemented
    P_DEFAULT = NotImplemented

    def __init__(self):
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
            val = [val[k] for k in self.pall]
        self._p0 = np.array(val)

    def get_p0(self, as_dict=True):
        if as_dict:
            return misc.make_dict(self.pall, self.p0)
        else:
            return self.p0

    def p0_is_set(self):
        return self._p0 is not None

    @property
    def p0_for_fit(self):
        """Fitted :py:attr:`p0`"""
        p0 = copy.deepcopy(self.p0)
        p0_for_fit = len(self.pfit) * [None]
        for k, i in self.pfit.items():
            p0_for_fit[i] = p0[self.pall[k]]
        return np.array(p0_for_fit)

    @property
    def popt(self):
        """All :py:attr:`popt` (non-fitted ones use :py:attr:`p0`)"""
        popt = self.p0.copy()
        if self._popt is not None:
            for k, i in self.pfit.items():
                popt[self.pall[k]] = self._popt[i]
        return np.array(popt)

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
        self._popt = val

    @property
    def pcov(self):
        return self._pcov

    @property
    def pcov_for_fit(self):
        return self._pcov

    @pcov_for_fit.setter
    def pcov_for_fit(self, val):
        self._pcov = val

    @property
    def pstd(self):
        # TODO: Get pall with dummy std for p0 only
        # TODO: Same for pcov
        return np.sqrt(np.diag(self.pcov))

    @property
    def pstd_for_fit(self):
        return np.sqrt(np.diag(self.pcov_for_fit))

    def get_pstd(self, as_dict=True):
        if as_dict:
            return misc.make_dict(self.pfit, self.pstd)
        else:
            return self.pstd

    def __getattr__(self, name):
        get_std = False
        if name[-4:] == "_std":
            get_std = True
            name = name[:-4]
        try:
            if get_std is True:
                return self.pstd[self.pall[name]]
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
        _p = copy.deepcopy(self.p0)
        _pall = self.pall
        _pfit = self.pfit
        _func = self._func

        def _fit_func(var, *p):
            for k_fit, i_fit in _pfit.items():
                _p[_pall[k_fit]] = p[i_fit]
            return _func(var, *_p).ravel()

        # Prepare fit data
        split_data = self._split_fit_data(*data)
        var_data, func_data = self._ravel_data(*split_data)
        p0 = self.p0_for_fit

        # Optimize parameters
        self.popt_for_fit, self.pcov_for_fit = scipy.optimize.curve_fit(
            _fit_func, var_data, func_data, p0=p0, **kwargs
        )
        self.psuccess = (
            not np.any(np.isnan(self.pcov_for_fit))
            and np.all(np.isfinite(self.pcov_for_fit))
        )
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
            FIXME: If two arrays are given, they are interpreted as
            `(var_data, func_data)`.
            TODO: Add err_data support.
        func_dim : `int`
            Only used for `SeriesData`.
            Dependent data dimension (index).

        Returns
        -------
        var_data : `np.ndarray`
            Independent data.
        func_data : `np.ndarray`
            Dependent data.
        """
        # Parse arguments
        if len(data) == 1:
            data = data[0]
        elif len(data) == 2:
            # FIXME: check for types
            return data
        else:
            raise NotImplementedError
        # Split data
        var_data, func_data = None, None
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
        return var_data, func_data

    @staticmethod
    def _ravel_data(var_data, func_data=None, _check_shape=True):
        """
        Serializes array-like (nD) data into series-like (1D) data.

        Parameters
        ----------
        var_data : `np.ndarray`
            Array-like independent data.
        func_data : `np.ndarray`
            Array-like dependent data.
        _check_shape : `bool`
            Flag whether to check `var_data` and `func_data`
            shape overlap.

        Returns
        -------
        var_data : `np.ndarray`
            Serialized independent data.
        func_data : `np.ndarray`
            Serialized dependent data.

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
        if var_data_ind == 0:
            var_data = var_data.ravel()
        else:
            var_data = var_data.reshape((var_data.shape[0], -1))
        return var_data, func_data