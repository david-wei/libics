import copy
import numpy as np

from scipy import interpolate

from libics.env import logging
from libics.core.data import types
from libics.core.util import misc


###############################################################################


class ArrayData(object):

    """
    Stores a multidimensional array and its scaling information (linear
    scaling and physical quantity).
    """

    POINTS = "POINTS"
    RANGE = "RANGE"
    LINSPACE = "LINSPACE"
    VAR_MODES = {POINTS, RANGE, LINSPACE}

    LOGGER = logging.get_loggger("libics.core.data.arrays.ArrayData")

    def __init__(self):
        # Data
        self.data_quantity = None
        self._data = np.empty(0)
        # Variable
        self.var_quantity = []
        self.var_mode = []
        # Mode: points
        self._points = []
        # Mode: range
        self._offset = []
        self._center = []
        self._step = []
        # Mode: linspace
        self._low = []
        self._high = []

    @staticmethod
    def _get_quantity(**kwargs):
        """
        Generates a quantity object from keyword arguments.

        Parameters
        ----------
        quantity : `types.Quantity`
            Directly specified quantity. Takes precedence
            over other methods.
        name, symbol, unit : `str`
            Parameters used to construct `types.Quantity` instance.
        """
        if "quantity" in kwargs:
            _quantity = misc.assume_construct_obj(
                kwargs["quantity"], types.Quantity
            )
        else:
            _quantity = types.Quantity()
            if "name" in kwargs:
                _quantity.name = kwargs["name"]
            if "symbol" in kwargs:
                _quantity.symbol = kwargs["symbol"]
            if "unit" in kwargs:
                _quantity.unit = kwargs["unit"]
        return _quantity

    def set_data_quantity(self, **kwargs):
        """
        Sets the data quantity object.

        See :py:meth:`_get_quantity`.
        """
        _quantity = self._get_quantity(**kwargs)
        self.data_quantity = _quantity

    def set_var_quantity(self, dim, **kwargs):
        """
        Sets the data quantity object.

        See :py:meth:`_get_quantity`.

        Parameters
        ----------
        dim : `int`
            Variable dimension to be set.
        """
        _quantity = self._get_quantity(**kwargs)
        self.var_quantity[dim] = _quantity

    def set_dim(self, dim, **kwargs):
        """
        Sets the numeric variable description for a dimension.

        Parameters
        ----------
        dim : `int`
            Variable dimension to be set.
        **kwargs
            A dimension can be defined by either of the three modes
            `POINTS, RANGE, LINSPACE`.
        ArrayData.POINTS
            `points` : `np.ndarray`
                1D array directly specifying the coordinates.
        ArrayData.RANGE
            `offset` : `float`
                Range offset (starting value).
            `center` : `float`
                Range center. Is ignored if `offset` is specified.
            `step` : `float`
                Range step (difference value).
        ArrayData.LINSPACE
            `low` : `float`
                Lower bound of linear spacing.
            `high` : `float`
                Upper bound of linear spacing.
        """
        # Mode: points
        if "points" in kwargs:
            self.var_mode[dim] = self.POINTS
            self._points[dim] = kwargs["points"]
            self._offset[dim] = None
            self._center[dim] = None
            self._step[dim] = None
            self._low[dim] = None
            self._high[dim] = None
        # Mode: range
        if ("offset" in kwargs or "center" in kwargs) and "step" in kwargs:
            self.var_mode[dim] = self.RANGE
            self._points[dim] = None
            if "offset" in kwargs:
                self._offset[dim] = kwargs["offset"]
                self._center[dim] = None
            else:
                self._offset[dim] = None
                self._center[dim] = kwargs["center"]
            self._step[dim] = kwargs["step"]
            self._low[dim] = None
            self._high[dim] = None
        # Mode: linspace
        if "low" in kwargs and "high" in kwargs:
            self.var_mode[dim] = self.LINSPACE
            self._points[dim] = None
            self._offset[dim] = None
            self._center[dim] = None
            self._step[dim] = None
            self._low[dim] = kwargs["low"]
            self._high[dim] = kwargs["high"]
        # Mode: unspecified
        else:
            self.var_mode[dim] = self.RANGE
            self._points[dim] = None
            self._offset[dim] = 0
            self._center[dim] = None
            self._step[dim] = 1
            self._low[dim] = None
            self._high[dim] = None

    def add_dim(self, *args):
        """
        Appends variable dimension(s) to the object.

        Parameters
        ----------
        *args : `dict` or `int`
            Dictionaries in the order they should be added.
            These dictionaries are used as ``kwargs`` for iteratively
            setting dimensions.
            An integer is equivalent to passing an this-integer-length
            list of empty dictionaries.

        Raises
        ------
        ValueError
            If arguments are invalid.
        """
        # Add nothing
        if len(args) == 0:
            return
        # Add empty dimensions
        if len(args) == 1 and isinstance(args[0], int):
            return self.add_dim(args[0] * [{}])
        # Invalid value handling
        for arg in args:
            if not isinstance(arg, dict):
                raise ValueError("invalid args: {:s}".format(str(args)))
        # Add dimensions
        self.var_mode.append(None)
        self._points.append(None)
        self._offset.append(None)
        self._center.append(None)
        self._step.append(None)
        self._low.append(None)
        self._high.append(None)
        # Set dimensions
        for i, arg in range(-len(args), 0):
            self.set_var_quantity(i, **arg)
            self.set_dim(i, **arg)

    def rmv_dim(self, *dims, num=None):
        """
        Removes variable dimension(s) from the object.

        Parameters
        ----------
        *dims : `int`
            Dimensions to be removed.
        num : `int`
            Removes `num` last dimensions. Is only applied
            if `dims` is not specified.
        """
        if len(dims) == 0 and isinstance(num, int):
            dims = np.arange(-num, 0)
        dims = np.sort(dims) % self.var_ndim
        for dim in reversed(dims):
            del self.var_mode[dim]
            del self._points[dim]
            del self._offset[dim]
            del self._center[dim]
            del self._step[dim]
            del self._low[dim]
            del self._high[dim]

    # ++++++++++
    # Conversion
    # ++++++++++

    def cv_index_to_quantity(self, ind, dim):
        """
        Converts an array index to a physical quantity value.

        Parameters
        ----------
        ind :
            Index to be converted.
        dim : int
            Dimension.

        Returns
        -------
        val :
            Physical quantity associated with given quantity.
        """
        # TODO:
        return self.offset[dim] + self.scale[dim] * ind

    def cv_quantity_to_index(self, val, dim,
                             round_index=True, incl_maxpoint=False):
        """
        Converts a physical quantity value to an array index.

        Parameters
        ----------
        val :
            Physical quantity value to be converted.
        dim : int
            Dimension.
        round_index : bool, optional
            Flag: whether to round to nearest index integer.
        incl_maxpoint: bool, optional
            Flag: whether to limit the returned index to be
            smaller or equal to the length of the array
            (i.e. whether to include stop in [start, stop]).

        Returns
        -------
        ind : int or float
            Index associated with given quantity.
            Data type depends on round_index parameter.

        Raises
        ------
        IndexError:
            If dim is out of bounds.
        """
        # TODO:
        val = max(val, self.offset[dim])
        if self.max[dim] is not None:
            val = min(val, self.max[dim])
            if val == self.max[dim] and not incl_maxpoint:
                val -= self.scale[dim]
        ind = (val - self.offset[dim]) / self.scale[dim]
        if round_index:
            ind = round(ind)
        return ind

    def cv_unit(self, dim, new_unit, op, val):
        """
        Converts the unit of a given dimension.

        Parameters
        ----------
        dim : int
            Dimension of conversion.
        new_unit : str
            New unit string.
        op : str or callable
            Performed operation as function or encoded as string.
            str:
                Supported string encoded operations include:
                "+", "-", "*", "/".
                Passed val parameter is used as second
                operand.
            callable:
                Call signature must be:
                `op(stored_value, val)`.
        val : int or float
            Value with which the operation is performed.
        """
        # TODO:
        if not callable(op):
            try:
                op = misc.operator_mapping[op]
            except(KeyError):
                return
        self.quantity[dim].unit = new_unit
        if dim < self.scale.get_dim() - 1:
            self.scale.offset[dim] = op(self.scale.offset[dim], val)
            self.scale.scale[dim] = op(self.scale.scale[dim], val)
            if self.scale.max[dim] is not None:
                self.scale.max[dim] = op(self.scale.max[dim], val)
        else:
            self.data = op(self.data, val)

    def get_points(self, dim):
        """
        Gets the variable points for the specified dimension.
        """
        if self.var_mode[dim] == self.POINTS:
            return self._points[dim]
        elif self.var_mode[dim] == self.RANGE:
            _step = self._step[dim]
            _range = self.shape[dim] * _step
            if self._offset[dim] is not None:
                _offset = self._offset[dim]
            else:
                _offset = self._center[dim] - _range / 2
            _stop = _offset + _range + 0.1 * _step
            return np.arange(_offset, _stop, _step)
        elif self.var_mode[dim] == self.LINSPACE:
            _num = self.shape[dim]
            return np.linspace(self._low[dim], self._high[dim], num=_num)
        else:
            raise ValueError("invalid mode: {:s}".format(str(self.var_mode)))

    def get_offset(self, dim):
        """
        Gets the variable offset for the specified dimension.
        """
        if self.var_mode[dim] == self.POINTS:
            self.LOGGER.warning(
                "getting variable offset on a dimension in POINTS mode"
            )
            return np.min(self._points[dim])
        elif self.var_mode[dim] == self.RANGE:
            if self._offset[dim] is not None:
                return self._offset[dim]
            else:
                _step = self._step[dim]
                _range = self.shape[dim] * _step
                _center = self._center[dim]
                return _center - _range / 2
        elif self.var_mode[dim] == self.LINSPACE:
            return self._low[dim]

    def get_center(self, dim):
        """
        Gets the variable center for the specified dimension.
        """
        if self.var_mode[dim] == self.POINTS:
            self.LOGGER.warning(
                "getting mean as variable center on a dimension in POINTS mode"
            )
            return np.mean(self._points[dim])
        elif self.var_mode[dim] == self.RANGE:
            if self._center[dim] is not None:
                return self._center[dim]
            else:
                _step = self._step[dim]
                _range = self.shape[dim] * _step
                _offset = self._offset[dim]
                return _offset + _range / 2
        elif self.var_mode[dim] == self.LINSPACE:
            return np.mean([self._low[dim], self._high[dim]])

    def get_step(self, dim):
        """
        Gets the variable step for the specified dimension.
        """
        if self.var_mode[dim] == self.POINTS:
            self.LOGGER.warning(
                "getting differential mean as variable step on a dimension "
                + "in POINTS mode"
            )
            _points = np.sort(self._points[dim])
            return np.mean(_points[1:] - _points[:-1])
        elif self.var_mode[dim] == self.RANGE:
            return self._step[dim]
        elif self.var_mode[dim] == self.LINSPACE:
            return (self._high[dim] - self._low[dim]) / (self.shape[dim] - 1)

    def get_low(self, dim):
        """
        Gets the variable low for the specified dimension.
        """
        if self.var_mode[dim] == self.POINTS:
            self.LOGGER.warning(
                "getting variable low on a dimension in POINTS mode"
            )
            return np.min(self._points[dim])
        elif self.var_mode[dim] == self.RANGE:
            if self._offset[dim] is not None:
                return self._offset[dim]
            else:
                _step = self._step[dim]
                _range = self.shape[dim] * _step
                _center = self._center[dim]
                return _center - _range / 2
        elif self.var_mode[dim] == self.LINSPACE:
            return self._low[dim]

    def get_high(self, dim):
        """
        Gets the variable high for the specified dimension.
        """
        if self.var_mode[dim] == self.POINTS:
            self.LOGGER.warning(
                "getting variable high on a dimension in POINTS mode"
            )
            return np.min(self._points[dim])
        elif self.var_mode[dim] == self.RANGE:
            _step = self._step[dim]
            _range = self.shape[dim] * _step
            if self._offset[dim] is not None:
                return self._offset[dim] + _range
            else:
                _center = self._center[dim]
                return _center + _range / 2
        elif self.var_mode[dim] == self.LINSPACE:
            return self._high[dim]

    # ++++++++++
    # Properties
    # ++++++++++

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def var_ndim(self):
        return len(self.var_mode)

    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        return len(self._data)

    @property
    def points(self):
        return [self.get_points(dim) for dim in range(self.ndim)]

    @property
    def offset(self):
        return [self.get_offset(dim) for dim in range(self.ndim)]

    @property
    def center(self):
        return [self.get_center(dim) for dim in range(self.ndim)]

    @property
    def step(self):
        return [self.get_step(dim) for dim in range(self.ndim)]

    @property
    def low(self):
        return [self.get_low(dim) for dim in range(self.ndim)]

    @property
    def high(self):
        return [self.get_high(dim) for dim in range(self.ndim)]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val
        diff_ndim = self.ndim - self.var_ndim
        if diff_ndim > 0:
            self.add_dim(diff_ndim)
        elif diff_ndim < 0:
            self.rmv_dim(num=np.abs(diff_ndim))

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def __str__(self):
        s = str(self.data_quantity)
        s += "Quantity:\n["
        for i in range(self.ndim):
            if i > 0:
                s += " "
            s += str(self.var_quantity[i])
            if i < self.ndim - 1:
                s += ",\n"
        s += "Var:\n["
        for i in range(self.ndim):
            if i > 0:
                s += " "
            if self.var_mode[i] == self.POINTS:
                s += "{:s}".format(str(self.get_points(i)))
            else:
                s += "range({:f}, {:f}, {:f})".format(
                    self.get_low(i), self.get_high(i), self.get_step(i)
                )
            if i < self.ndim - 1:
                s += ",\n"
        s += "]\n"
        s += "Data:\n"
        s += "{:s}".format(str(self.data))
        return s

    # +++++++++++++
    # Interpolation
    # +++++++++++++

    def __call__(self, var, mode="nearest", extrapolation=False):
        """
        Parameters
        ----------
        var : list(float) or list(np.ndarray(float))
            Requested variables for which the functional value is
            obtained. The variable format must be a list of each
            variable dimension (typically a flattened ij-indexed
            meshgrid). Supports higher dimensional meshgrids, but
            is discouraged because of index ambiguity.
            Shape:
                (data dimension, *) where * can be any scalar
                or array.
        mode : str
            "nearest": Value of nearest neighbour.
            "linear": Linear interpolation.
        extrapolation : bool or float
            True: Performs nearest/linear extrapolation.
            False: Raises ValueError if extrapolated value is
                   requested.
            float: Used as extrapolation fill value.

        Returns
        -------
        func_val : np.ndarray(float)
            Functional values of given variables.
            Shape: var.shape[1:].

        Raises
        ------
        ValueError
            See parameter extrapolation.

        See Also
        --------
        scipy.interpolate.interpn
        """
        # Convert to seriesdata-structure-like
        shape = None
        dim = self.ndim
        if dim == 1:
            if np.isscalar(var) or len(var) != 1:
                var = [var]
            var = np.array(var)
        if not np.isscalar(var[0]):
            shape = var[0].shape
            var = var.reshape((dim, var.size // dim))
            var = np.moveaxis(var, 0, -1)
        # Set up interpolation variables
        points = self.points
        values = self.data
        xi = var
        method = mode
        bounds_error = None
        fill_value = None
        if extrapolation is True:
            bounds_error = False
            fill_value = None
        elif extrapolation is False:
            bounds_error = True
        else:
            bounds_error = False
            fill_value = extrapolation
        # Interpolation
        func_val = interpolate.interpn(
            points, values, xi, method=method, bounds_error=bounds_error,
            fill_value=fill_value
        )
        if shape is not None:
            func_val = func_val.reshape(shape)
        return func_val

    # ++++++++++++++++++++++
    # Combination operations
    # ++++++++++++++++++++++

    def cmp_attr(self, other):
        return self.chk_attr() and other.chk_attr()

    def cmp_unit(self, other):
        """
        Compares whether the units of two ArrayData objects are identical.
        """
        for dim, quantity in enumerate(self.scale.quantity):
            if not quantity.unit == other.quantity[dim].unit:
                return False
        return True

    def cmp_shape(self, other):
        """
        Compares whether the shapes of two ArrayData objects are identical.
        """
        return self.data.shape == other.data.shape

    def cmp_scale(self, other):
        """
        Compares whether the scales of two ArrayData objects are identical.
        """
        return np.all(np.isclose(self.scale.scale, other.scale.scale))

    def cmp_offset(self, other):
        """
        Compares whether the offsets of two ArrayData objects are identical.
        """
        return np.all(np.isclose(self.scale.offset, other.scale.offset))

    def cmp_necessary(self, other):
        """
        Performs necessary pre-combination comparison checks.
        """
        return (
            self.cmp_attr(other)
            and self.cmp_unit(other)
        )

    def chk_commensurable(self, other):
        """
        Checks whether two ArrayData objects are commensurable, i.e. can be
        combined based on arrays (without index interpolation by quantity).
        """
        scale_multiple = (
            (np.array(other.scale.offset) - np.array(self.scale.offset))
            / np.array(other.scale.scale)
        )
        return (
            self.cmp_scale(other)
            and np.all(np.isclose(scale_multiple, np.round(scale_multiple)))
        )

    def get_common_rect(self, other):
        """
        Gets the common offset and max values.

        Returns
        -------
        offset : list
            Offset of common rectangle.
        max_ : list
            Max of common rectangle (excluding index).
        self_offset_index
        self_max_index
        other_offset_index
        other_max_index
        """
        offset, max_ = [], []
        for i, _ in enumerate(self.scale.offset):
            offset.append(max(self.scale.offset[i], other.scale.offset[i]))
            max_.append(min(self.scale.max[i], other.scale.max[i]))
        return (
            offset, max_,
            self.scale.get_index_by_quantity(offset),
            self.scale.get_index_by_quantity(max_),
            other.scale.get_index_by_quantity(offset),
            other.scale.get_index_by_quantity(max_)
        )

    def get_common_obj(self, other, op, in_place=False):
        """
        Creates the common result after operation.

        The data space is chosen to be common to both ArrayData objects.
        The given operation is performed to combine both.

        Parameters
        ----------
        op : callable
            Function signature:
            op(numpy.ndarray, numpy.ndarray) -> numpy.ndarray.
        in_place : bool
            Flag whether to store result in same instance.

        Returns
        -------
        obj : ArrayData
            Processed object.

        Raises
        ------
        ValueError:
            If the two ArrayData objects cannot be combined.
        """
        obj = self

        # Non-homogeneous operation
        if not isinstance(other, ArrayData):
            if not in_place:
                obj = copy.deepcopy(self)
            obj.data = op(self.data, other)
            return obj

        # Check homogeneous operation validity
        if not self.cmp_necessary(other):
            raise ValueError("Invalid necessary values")
        # Initialize minimal ArrayData object
        if not in_place:
            obj = ArrayData()
            obj.scale = copy.deepcopy(self.scale)
            (
                obj.scale.offset, obj.scale.max,
                si_offset, si_max,
                oi_offset, oi_max
            ) = self.get_common_rect(other)
            obj.data = self.data[
                [slice(si_offset[i], si_max[i]) for i in range(len(si_offset))]
            ]

        # Perform operation on data array
        # TODO: Add efficient on-array operations for integer-shifted and
        #       commensurately scaled data (m + n -> f * m + n + n')
        if self.chk_commensurable():
            obj.data = op(
                obj.data,
                other.data[
                    [slice(oi_offset[i], oi_max[i])
                     for i in range(len(oi_offset))]
                ]
            )
        # Interpolate between data arrays
        else:
            it = np.nditer(obj.data, flags=["multi_index"])
            while not it.finished:
                ind = other.get_index_by_quantity(
                    obj.scale.cv_index_to_quantity(it.multi_index)
                )
                obj.data[it.multi_index] = op(
                    obj.data[it.multi_index],
                    other.data[ind]
                )
                it.iternext()
        return obj

    def get_copy_obj(self, op):
        """
        Creates a copy of self object and applies given operation on it.

        Parameters
        ----------
        op : callable
            Function signature:
            op(numpy.ndarray) -> numpy.ndarray

        Returns
        -------
        obj : ArrayData
        """
        obj = copy.deepcopy(self)
        obj.data = op(obj.data)
        return obj

    # ++++ Arithmetics ++++++++++++

    def __add__(self, other):
        return self.get_common_obj(other, np.add)

    def __sub__(self, other):
        return self.get_common_obj(other, np.subtract)

    def __mul__(self, other):
        return self.get_common_obj(other, np.multiply)

    def __truediv__(self, other):
        return self.get_common_obj(other, np.true_divide)

    def __floordiv__(self, other):
        return self.get_common_obj(other, np.floor_divide)

    def __mod__(self, other):
        return self.get_common_obj(other, np.mod)

    def __pow__(self, other):
        return self.get_common_obj(other, np.power)

    # ++++ Comparisons +++++++++++

    def __lt__(self, other):
        return np.all(self.get_common_obj(other, np.less).data)

    def __le__(self, other):
        return np.all(self.get_common_obj(other, np.less_equal).data)

    def __eq__(self, other):
        try:
            return np.all(self.get_common_obj(other, np.equal).data)
        except ValueError:
            return False

    def __ne__(self, other):
        try:
            return np.all(self.get_common_obj(other, np.not_equal).data)
        except ValueError:
            return True

    def __ge__(self, other):
        return np.all(self.get_common_obj(other, np.greater_equal).data)

    def __gt__(self, other):
        return np.all(self.get_common_obj(other, np.greater).data)

    # ++++ Binary operations +++++

    def __and__(self, other):
        return self.get_common_obj(other, np.bitwise_and)

    def __or__(self, other):
        return self.get_common_obj(other, np.bitwise_or)

    def __xor__(self, other):
        return self.get_common_obj(other, np.bitwise_xor)

    # ++++ Unary operations ++++++

    def __abs__(self):
        return self.get_copy_obj(np.abs)

    def __round__(self):
        return self.get_copy_obj(np.round)

    def __invert__(self):
        return self.get_copy_obj(np.invert)

    def __lshift__(self):
        return self.get_copy_obj(np.left_shift)

    def __rshift__(self):
        return self.get_copy_obj(np.right_shift)

    def __neg__(self):
        return self.get_copy_obj(np.negative)

    def __bool__(self):
        return isinstance(self.data, np.ndarray) and len(self.data) > 0

    # ++++ Type conversions ++++++

    def __int__(self):
        self.data = self.data.astype(int)
        return self

    def __float__(self):
        self.data = self.data.astype(float)
        return self

    def __complex__(self):
        self.data = self.data.astype(complex)
        return self

    # ++++ In-place arithmetics ++

    def __iadd__(self, other):
        return self.get_common_obj(other, np.add, in_place=True)

    def __isub__(self, other):
        return self.get_common_obj(other, np.subtract, in_place=True)

    def __imul__(self, other):
        return self.get_common_obj(other, np.multiply, in_place=True)

    def __itruediv__(self, other):
        return self.get_common_obj(other, np.true_divide, in_place=True)

    def __ifloordiv__(self, other):
        return self.get_common_obj(other, np.floor_divide, in_place=True)

    def __imod__(self, other):
        return self.get_common_obj(other, np.mod, in_place=True)

    def __ipow__(self, other):
        return self.get_common_obj(other, np.power, in_place=True)

    def __iand__(self, other):
        return self.get_common_obj(other, np.bitwise_and, in_place=True)

    def __ior__(self, other):
        return self.get_common_obj(other, np.bitwise_or, in_place=True)

    def __ixor__(self, other):
        return self.get_common_obj(other, np.bitwise_xor, in_place=True)

    # ++++ Numpy universal functions

    def __array_ufunc__(self, ufunc, method, i, *inputs, **kwargs):
        # Convert ArrayData inputs into np.ndarray inputs
        inputs = tuple([(it.data if isinstance(it, ArrayData) else it)
                        for it in inputs])
        # Declare output object
        obj = None
        # Allocated memory already declared
        if "out" in kwargs.keys():
            obj = kwargs["out"]
            kwargs["out"] = tuple([
                (it.data if isinstance(it, ArrayData) else it) for it in obj
            ])
            self.data.__array_ufunc__(ufunc, method, i, *inputs, **kwargs)
        # Construct new object
        else:
            obj = copy.deepcopy(self)
            obj.data = self.data.__array_ufunc__(
                ufunc, method, i, *inputs, **kwargs
            )
        # Return ArrayData object with data as calculated by the ufunc
        return obj


###############################################################################


class SeriesData(object):

    def __init__(self, dim=0, pkg_name="libics", cls_name="SeriesData"):
        super().__init__(pkg_name=pkg_name, cls_name=cls_name)
        self.init(dim)

    def init(self, dim):
        self._data = np.empty((0, 0))
        self.quantity = dim * [types.Quantity()]

    def add_dim(self, quantity=None, name="N/A", symbol=None, unit=None):
        """
        Appends a dimension to the object.

        Parameters
        ----------
        quantity : types.Quantity or None
            Quantity. If specified, overwrites name and unit.
        name : str
            Quantity name.
        symbol : str or None
            Quantity symbol.
        unit : str or None
            Quantity unit.
        """
        if quantity is not None:
            self.quantity.append(quantity)
        else:
            self.quantity.append(types.Quantity(
                name=name, symbol=symbol, unit=unit
            ))

    def get_dim(self):
        """
        Gets the data dimension (incl. actual data).
        """
        return len(self.quantity)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        if len(val) == len(self.quantity):
            self._data = val
        else:
            raise ValueError("invalid data shape ({:d}/{:d})"
                             .format(len(val), len(self.quantity)))

    @property
    def loc(self, key):
        """
        Get data by numpy index addressing [entry, variable].
        """
        return self.data[key]

    @loc.setter
    def loc(self, key, val):
        """
        Set data by numpy index addressing [entry, variable].
        """
        self.data[key] = val

    def __getitem__(self, key):
        """
        Get data by variable index addressing [variable].

        Examples
        --------
        >>> sd = SeriesData()
        >>> sd.data = np.arange(30).reshape((3, 10))
        >>> sd.add_dimension(name="x")      # index 0
        >>> sd.add_dimension(name="y")      # index 1
        >>> sd.add_dimension(name="val")    # index 2
        >>> sd[1]
        array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        >>> sd[0:2]
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        >>> sd[0, 2]
        array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
               [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]])
        """
        if isinstance(key, tuple) or isinstance(key, list):
            return self.data[[key]]
        else:
            return self.data[key]

    def __setitem__(self, key, val):
        """
        Set data by variable index addressing [variable].
        """
        if isinstance(key, tuple) or isinstance(key, list):
            self.data[[key]] = val
        else:
            self.data[key] = val
