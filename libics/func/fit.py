import abc

import numpy as np
from scipy import optimize

from libics.cfg import err as ERR
from libics.util import misc


###############################################################################


class FitParamBase(object, abc.ABC):

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
    """

    def __init__(self):
        self.func = misc.do_nothing()   # Fit function
        self.param = []                 # [parameter value]
        self.std = []                   # [standard deviation]
        self.cov = [[]]                 # [[covariance]]
        self.ind_to_name = {}           # {index: parameter name}
        self.name_to_ind = {}           # {parameter name: index}
        self._shape = None              # Shape used for (un-) ravel

    def set_param_map(self, param_list):
        """
        Defines the parameter names and enables get by string.

        Parameters
        ----------
        param_list : list(str)
            List of unique parameter ID strings in the order
            of the parameters.
        """
        for i in range(len(param_list)):
            self.ind_to_name[i] = param_list[i]
            self.name_to_ind[param_list[i]] = i

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
    def find_init_param(self, data):
        """
        Calculates the initial parameters and saves them as parameters.

        Parameters
        ----------
        data : np.ndarray
            Data to be fitted.
        """
        raise NotImplementedError

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
        self.std = np.sqrt(np.diagonal(self.cov))

    def __call__(self, var):
        """
        Calls and evaluates the function for the given variables.

        Parameters
        ----------
        var : np.ndarray
            Independent variable passed for evaluation.

        Returns
        -------
        res : np.ndarray
            Evaluation result.
        """
        return self.func(var, *self.param)

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
