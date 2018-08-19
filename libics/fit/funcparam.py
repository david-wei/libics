###############################################################################


class FitParamBase(object):

    """
    Base class for fitting. Contains fitting parameters and function calls.
    Includes uncertainties.
    """

    def __init__(self):
        self.params = []    # [(parameter value, standard deviation)]
        self.cov = [[]]     # [[covariance]]
        self.ind_to_name = {}  # {index: parameter name}
        self.name_to_ind = {}  # {parameter name: index}

    def set_param_map(self, param_list):
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
            return self.params[key]
