import abc

import numpy as np
from scipy import optimize

from libics.cfg import err as ERR
from libics.data import arraydata, seriesdata
from libics.util import misc


###############################################################################


def split_fit_data(data, func_dim=-1):
    """
    Splits the given data into independent and dependent data, as required
    by the fit class.

    Parameters
    ----------
    data : arraydata.ArrayData or seriesdata.SeriesData or np.ndarray
        Data to be split.
    func_dim : int
        Only used for seriesdata.SeriesData.
        Dependent data dimension (index).

    Returns
    -------
    var_data : np.ndarray
        Independent data.
    func_data : np.ndarray
        Dependent data.
    """
    var_data, func_data = None, None
    if isinstance(data, seriesdata.SeriesData):
        func_data = data.data[func_dim]
        var_data = np.concatenate(
            (data.data[0:func_dim], data.data[func_dim + 1:])
        )
    else:
        if isinstance(data, np.ndarray):
            func_data = data
        elif isinstance(data, arraydata.ArrayData):
            func_data = data.data
        var_data = np.indices(func_data.shape)
        if isinstance(data, arraydata.ArrayData):
            for i in range(len(var_data)):
                scale = data.scale.scale[i]
                offset = data.scale.offset[i]
                var_data[i] = var_data[i] * scale + offset
    return var_data, func_data


###############################################################################


class FitParamBase(abc.ABC):

    """
    Base class for fitting. Contains fitting parameters and function calls.
    Includes uncertainties.

    Parameters
    ----------
    func : callable
        Fit function with call signature func(var, *param).
            var : scalar, np.ndarray
                If array-like, the first dimension must accept
                different independent parameters, i.e. (x, y, z, ...).
            param : scalar
                Fit parameters in the order as stored in this class.
    param : np.ndarray(1) or int
        If np.ndarray: used as current parameters.
        If int: sets dimension of parameters (i.e. constructs an
                np.ndarray with `param` ones as elements).
    cov : np.ndarray(2) or np.ndarray(1) or scalar
        If np.ndarray(2): used as current covariance matrix.
        If np.ndarray(1): used as current standard deviation. Off-diagonal
                          elements are set to 0.
        If scalar: covariance matrix filled with given values.
    param_names : list(str) or None
        List of unique parameter names.
        If None, default names are index numbers.
    """

    def __init__(
        self, func, param, cov=0, param_names=None
    ):
        # Function
        self.func = func
        # Parameters
        if np.isscalar(param):
            self.param = np.ones(np.round(param), dtype=float)
        else:
            self.param = np.array(param, dtype=float)
        # Covariance and standard deviation
        if np.isscalar(cov):
            self.cov = np.full(
                (len(self.param), len(self.param)), cov, dtype=float
            )
        else:
            cov = np.array(cov, dtype=float)
            if cov.ndim == 1:
                self.cov = np.diag(cov)**2
            elif cov.ndim == 2:
                self.cov = cov
            else:
                raise ValueError("invalid covariance")
        # Parameter names
        if param_names is None:
            param_names = [str(i) for i in range(len(self.param))]
        self.ind_to_name = {}           # {index: parameter name}
        self.name_to_ind = {}           # {parameter name: index}
        self.set_param_map(param_names)
        # Internal
        self._shape = None              # Shape used for (un-) ravel

    def set_param_map(self, param_names):
        """
        Defines the parameter names and enables get by string.

        Parameters
        ----------
        param_names : list(str)
            List of unique parameter ID strings in the order
            of the parameters.
        """
        self.ind_to_name = misc.make_dict(np.arange(len(param_names)),
                                          param_names)
        self.name_to_ind = misc.reverse_dict(self.ind_to_name)

    @property
    def std(self):
        return np.sqrt(np.diagonal(self.cov))

    def __getitem__(self, key):
        """
        If passing a single value, the respective parameter (value, standard
        deviation) is returned. If passing a 2-tuple, the respective covariance
        is returned. Passed values can be either the integer index or the name
        of the parameter.
        Parameters are not checked for validity.

        Parameters
        ----------
        key : `int` or `str` or `tuple(int or str)`
            `int` or `str`: Index or name of parameter.
            `tuple(int or str)`: Indices or names of covariance.

        Returns
        -------
        param or cov : `float`
            Parameter (value, standard deviation) or covariance.
        """
        # If tuple as key, return covariance
        if type(key) == tuple:
            for i in range(len(key)):
                if type(key[i]) == str:
                    key[i] = self.name_to_ind[key[i]]
            return self.cov[key[0]][key[1]]
        # If single value as key, return parameter
        else:
            if type(key) == str:
                key = self.name_to_ind[key]
            return self.param[key]

    @abc.abstractmethod
    def find_init_param(self, var_data, func_data):
        """
        Calculates the initial parameters and saves them as parameters.

        Parameters
        ----------
        var_data : np.ndarray
            Independent variable data to be fitted.
            The first dimension must correspond to the function variable
            input.
        func_data : np.ndarray
            Dependent variable data (functional values) to be fitted.
        """
        self.param = np.ones_like(self.param, dtype=float)

    def set_init_param(self, param):
        """
        Sets initial parameters and checks these for validity.

        Parameters
        ----------
        param : np.ndarray
            Initial parameters used for fit.

        Raises
        ------
        ValueError
            If parameter is invalid.
        """
        if np.isscalar(param):
            param = np.array(param)
        if param.ndim != self.param.ndim or param.size != self.param.size:
            raise ValueError("invalid parameters")
        self.param = param

    def find_fit(self, var_data, func_data, **kwargs):
        """
        Calculates the parameter fit using the parameter values
        as initial parameters.

        Parameters
        ----------
        var_data : np.ndarray
            Independent variable data to be fitted.
            The first dimension must correspond to the function variable
            input.
        func_data : np.ndarray
            Dependent variable data (functional values) to be fitted.
        kwargs
            Keyword arguments passed to scipy.optimize.curve_fit, i.a.:
            sigma, absolute_sigma, check_finite, bounds, method, jac.
        """
        var_data, func_data = self._ravel_data(var_data, func_data)
        self.param, self.cov = optimize.curve_fit(
            self.func, var_data, func_data, p0=self.param, **kwargs
        )

    def __call__(self, var, *args, **kwargs):
        """
        Calls and evaluates the function for the given variables.

        Parameters
        ----------
        var : np.ndarray
            Independent variable passed for evaluation.
        args, kwargs
            (Keyword) arguments which are passed to the function.
            If given, the stored parameters are overwritten.

        Returns
        -------
        res : np.ndarray
            Evaluation result.
        """
        p = args if len(args) > 0 else self.param
        return self.func(var, *p, **kwargs)

    def _ravel_data(self, var_data, func_data=None):
        """
        Serializes array-like (nD) data into series-like (1D) data.

        Parameters
        ----------
        var_data : np.ndarray
            Array-like independent data.
        func_data : np.ndarray
            Array-like dependent data.

        Returns
        -------
        var_data : np.ndarray
            Serialized independent data.
        func_data : np.ndarray
            Serialized dependent data.

        Raises
        ------
        ERR.INVAL_STRUCT_NUMPY
            If data dimensions are invalid.
        """
        var_data = np.array(var_data)
        var_data_ind = 1 if var_data.ndim > 1 else 0
        if func_data is not None:
            func_data = np.array(func_data)
            ERR.assertion(ERR.INVAL_STRUCT_NUMPY,
                          var_data.shape[var_data_ind:] == func_data.shape,
                          description="invalid fit data dimensions")
            func_data = func_data.ravel()
        if var_data_ind == 0:
            self._shape = var_data.shape
            var_data = var_data.ravel()
        else:
            self._shape = var_data[var_data_ind:].shape
            var_data = var_data.reshape(
                (var_data.shape[0], np.prod(self._shape))
            )
        return var_data, func_data
