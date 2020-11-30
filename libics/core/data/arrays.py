import copy
import numpy as np
from scipy import interpolate

from libics.env import logging
from libics.core.data import types
from libics.core.util import misc


###############################################################################
# ArrayData
###############################################################################


class ArrayData(object):

    """
    Stores a multidimensional array and its scaling information (linear
    scaling and physical quantity).

    Usage of this class is described below.

    Object creation:

    * Instantiate an `ArrayData` object.
    * Optional: Define the dimensions of the data using :py:meth:`add_dim`
      by specifying the metadata and scaling behavior.
    * Set the data using the :py:attr:`data` attribute. Default dimensions
      are added or removed if they are not commensurate.
    * Alternative: This class can also be used to specify the variables
      (i.e. array indices) only, metadata and scaling behavior are set
      accordingly. Instead of setting the data, the attribute
      :py:attr`var_shape` can be set.

    Object modification:

    * Metadata can be also added after data assignment. This is done using the
      :py:meth:`set_dim` method.
    * The numeric metadata mode (i.e. the scaling behaviour) can be modified
      using :py:meth:`mod_dim`.
    * The metadata can be numerically changed by computing an arithmetic
      operations using :py:meth:`comp_dim`.
    * The object supports common unary and binary operations, as well as
      vectorized `numpy` ufuncs.

    Object properties:

    * Numeric metadata can be extracted (depending on the metadata mode).
    * Upon calling the object with some given variables, an interpolated
      result is returned.

    Subclassing or class modification:

    * Follow the convention to declare all instance attributes within the
      constructor.
    * Remember to add these attribute names to :py:meth:`copy_var`.
    """

    POINTS = "POINTS"
    RANGE = "RANGE"
    LINSPACE = "LINSPACE"
    VAR_MODES = {POINTS, RANGE, LINSPACE}

    LOGGER = logging.get_logger("libics.core.data.arrays.ArrayData")

    def __init__(self):
        # -------------------
        # Instance attributes
        # -------------------
        # Data
        self.data_quantity = None
        self.set_data_quantity()
        self._data = np.empty(0)
        self._placeholder_shape = tuple()
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

    __LIBICS_IO__ = True
    SER_KEYS = {
        "data_quantity", "data", "var_quantity", "var_mode",
        "_points", "_offset", "_center", "_step", "_low", "_high"
    }

    def attributes(self):
        """Implements :py:meth:`libics.core.io.FileBase.attributes`."""
        return {k: getattr(self, k) for k in self.SER_KEYS}

    # ++++
    # Data
    # ++++

    def set_data_quantity(self, **kwargs):
        """
        Sets the data quantity object.

        Parameters
        ----------
        **kwargs
            See :py:func:`assume_quantity`.
        """
        _quantity = assume_quantity(**kwargs)
        self.data_quantity = _quantity

    # +++++++++++++++++++++++++++++++
    # Variable (dimension management)
    # +++++++++++++++++++++++++++++++

    def set_var_quantity(self, dim, **kwargs):
        """
        Sets the data quantity object.

        Parameters
        ----------
        dim : `int`
            Variable dimension to be set.
        **kwargs
            See :py:func:`assume_quantity`.
        """
        _quantity = assume_quantity(**kwargs)
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
        elif ("offset" in kwargs or "center" in kwargs) and "step" in kwargs:
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
        elif "low" in kwargs and "high" in kwargs:
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

    def mod_dim(self, *args):
        """
        Modifies the numeric variable description mode (if possible).

        Parameters
        ----------
        *args : `ArrayData.VAR_MODES` or `(int, ArrayData.VAR_MODES)`
            If two parameters are given, they are interpreted as
            `(dim, mode)`.
            If one parameter is given, changes all dimensions
            to the given `mode`.
        dim : `int`
            Variable dimension to be changed.
        mode : `ArrayData.VAR_MODES`
            Mode to be changed to.

        Raises
        ------
        ValueError
            If mode change is not possible. This happens mainly for
            conversion from `POINT` mode where the points are not
            linearly spaced.
            If `mode` is invalid.
        """
        # Loop over all dimensions
        if len(args) == 1:
            mode = args[0]
            for dim in range(self.ndim):
                self.mod_dim(dim, mode)
            return
        # Error handling
        elif len(args) > 2:
            raise ValueError("invalid argument ({:s})".format(str(args)))
        # Change dimension
        dim, mode = args
        if mode not in self.VAR_MODES:
            raise ValueError("invalid variable mode ({:s})".format(str(mode)))
        if self.var_mode[dim] in [self.RANGE, self.LINSPACE]:
            if mode == self.POINTS:
                self.set_dim(dim, points=self.get_points(dim))
            elif mode == self.RANGE:
                self.set_dim(
                    dim, offset=self.get_offset(dim), step=self.get_step(dim)
                )
            elif mode == self.LINSPACE:
                self.set_dim(
                    dim, low=self.get_low(dim), high=self.get_high(dim)
                )
        elif self.var_mode[dim] == self.POINTS:
            _p = self.get_points(dim)
            _dp = _p[1] - _p[0]
            if not np.allclose(_p[1:] - _p[:-1], _dp):
                raise ValueError("could not convert from POINTS to {:s} mode"
                                 .format(str(mode)))
            if mode == self.RANGE:
                self.set_dim(dim, offset=_p[0], step=_dp)
            elif mode == self.LINSPACE:
                self.set_dim(dim, low=_p[0], high=_p[-1])

    def comp_dim(self, dim, op, other=None, rev=False):
        """
        Performs an arithmetic operation on the specified variable dimension.

        Parameters
        ----------
        dim : `int`
            Variable dimension to be changed.
        op : `str` or `callable`
            Operator string. Mapped by `data.types.BINARY_OPS_NUMPY`
            and `data.types.UNARY_OPS_NUMPY`.
            Can also be a function having a binary or unary call signature.
        other : `numeric` or `None`
            Second operand for binary operations or `None` for unary ones.
        rev : `bool`
            Flag whether to use `other` as first operand, i.e. as
            `op(other, self)` instead of `op(self, other)`.

        Raises
        ------
        KeyError
            If `op` is invalid.

        Notes
        -----
        For the variable modes `"RANGE", "LINSPACE"`, it is assumed
        that the operation is an affine transformation.
        """
        # Unary operations
        if other is None:
            if isinstance(op, str):
                op = types.UNARY_OPS_NUMPY[op]
            if self.var_mode[dim] == self.POINTS:
                self._points[dim] = op(self._points[dim])
            elif self.var_mode[dim] == self.RANGE:
                s = self._step[dim]
                if self._center[dim] is None:
                    o = self._offset[dim]
                    self._offset[dim] = op(o)
                    self._step[dim] = op(o + s) - self._offset[dim]
                elif self._offset[dim] is None:
                    c = self._center[dim]
                    self._center[dim] = op(c)
                    self._step[dim] = op(c + s) - self._center[dim]
            elif self.var_mode[dim] == self.LINSPACE:
                self._low[dim] = op(self._low[dim])
                self._high[dim] = op(self._high[dim])
        # Binary operations
        else:
            if isinstance(op, str):
                op = types.BINARY_OPS_NUMPY[op]
            if rev:
                def _op(x, y):
                    return op(y, x)
            else:
                _op = op
            if self.var_mode[dim] == self.POINTS:
                self._points[dim] = _op(self._points[dim], other)
            elif self.var_mode[dim] == self.RANGE:
                s = self._step[dim]
                if self._center[dim] is None:
                    o = self._offset[dim]
                    self._offset[dim] = _op(o, other)
                    self._step[dim] = _op(o + s, other) - self._offset[dim]
                elif self._offset[dim] is None:
                    c = self._center[dim]
                    self._center[dim] = _op(c, other)
                    self._step[dim] = _op(c + s, other) - self._center[dim]
            elif self.var_mode[dim] == self.LINSPACE:
                self._low[dim] = _op(self._low[dim], other)
                self._high[dim] = _op(self._high[dim], other)

    def add_dim(self, *args, **kwargs):
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
        **kwargs
            Keyword dictionary is interpreted as if a single dictionary
            with the `kwargs` as items was passed.
            If any `*args` are given, `**kwargs` are ignored.

        Raises
        ------
        ValueError
            If arguments are invalid.
        """
        # Handle kwargs
        if len(args) == 0:
            if len(kwargs) == 0:
                return
            else:
                args = [kwargs]
        # Handle multiple dimensions
        if len(args) == 1 and isinstance(args[0], int):
            args = args[0] * [{}]
        if len(args) > 1 and isinstance(args[0], dict):
            for arg in args:
                self.add_dim(arg)
            return
        # Handle single dimension
        arg = args[0]
        if not isinstance(arg, dict):
            raise ValueError("invalid argument: {:s}".format(str(arg)))
        # Add dimensions
        self.var_quantity.append(None)
        self.var_mode.append(None)
        self._points.append(None)
        self._offset.append(None)
        self._center.append(None)
        self._step.append(None)
        self._low.append(None)
        self._high.append(None)
        # Set dimensions
        self.set_var_quantity(-1, **arg)
        self.set_dim(-1, **arg)

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
            del self.var_quantity[dim]
            del self.var_mode[dim]
            del self._points[dim]
            del self._offset[dim]
            del self._center[dim]
            del self._step[dim]
            del self._low[dim]
            del self._high[dim]

    # ++++++
    # Getter
    # ++++++

    def get_points(self, dim):
        """
        Gets the variable points for the specified dimension.
        """
        if self.var_mode[dim] == self.POINTS:
            return self._points[dim]
        elif self.var_mode[dim] == self.RANGE:
            _step = self._step[dim]
            _range = (self.var_shape[dim] - 1) * _step
            if self._offset[dim] is not None:
                _offset = self._offset[dim]
            else:
                _offset = self._center[dim] - _range / 2
            _stop = _offset + _range + 0.1 * _step
            return np.arange(_offset, _stop, _step)
        elif self.var_mode[dim] == self.LINSPACE:
            _num = self.var_shape[dim]
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
            return (
                (self._high[dim] - self._low[dim]) / (self.var_shape[dim] - 1)
            )

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
                _range = self.var_shape[dim] * _step
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
            _range = self.var_shape[dim] * _step
            if self._offset[dim] is not None:
                return self._offset[dim] + _range
            else:
                _center = self._center[dim]
                return _center + _range / 2
        elif self.var_mode[dim] == self.LINSPACE:
            return self._high[dim]

    def get_bins(self, dim):
        """
        Gets the bins of variable points for the specified dimension.

        The returned bins (array of size `len(points) + 1`) refer to the
        centers between adjacent points.
        Edge bins (where one side is undefined) use the other side to form
        symmetric bins around the edge points.
        """
        _points = self.get_points(dim)
        _bins = np.empty(len(_points) + 1, dtype=float)
        _bins[1:-1] = (_points[:-1] + _points[1:]) / 2
        _bins[0] = 2 * _points[0] - _bins[1]
        _bins[-1] = 2 * _points[-1] - _bins[-2]
        return _bins

    def max(self):
        return np.max(self.data)

    def min(self):
        return np.min(self.data)

    def mean(self):
        return np.mean(self.data)

    def std(self):
        return np.std(self.data)

    def get_var_meshgrid(self, indexing="ij"):
        """
        Creates a `numpy` meshgrid for the variable dimensions.

        Parameters
        ----------
        indexing : `"ij"` or `"xy"`
            See :py:func:`numpy.meshgrid`.

        Returns
        -------
        mg : `np.ndarray`
            Meshgrid as `numpy` array.
        """
        return np.array(np.meshgrid(*self.points, indexing=indexing))

    def get_var_meshgrid_bins(self, indexing="ij"):
        """
        Creates a `numpy` meshgrid for the bins of the variable dimensions.

        Parameters
        ----------
        indexing : `"ij"` or `"xy"`
            See :py:func:`numpy.meshgrid`.

        Returns
        -------
        mg : `np.ndarray`
            Meshgrid of bins as `numpy` array.
        """
        return np.array(np.meshgrid(*self.bins, indexing=indexing))

    # ++++++++++
    # Properties
    # ++++++++++

    @property
    def size(self):
        """Data array size."""
        return self._data.size

    @property
    def ndim(self):
        """Data array ndim."""
        if self._data.shape == (0,):
            return 0
        else:
            return self._data.ndim

    @property
    def var_ndim(self):
        """Variable ndim."""
        return len(self.var_mode)

    @property
    def total_ndim(self):
        """Total ndim (variable and data dimension)."""
        return self.ndim + 1

    @property
    def shape(self):
        """Data array shape."""
        return self._data.shape

    @property
    def var_shape(self):
        """Variable (placeholder) shape. Is updated upon setting `data`."""
        if self.ndim == self.var_ndim:
            return self.shape
        else:
            return self._placeholder_shape

    @var_shape.setter
    def var_shape(self, val):
        self._placeholder_shape = tuple(val)

    def __len__(self):
        """Length of data."""
        return len(self._data)

    @property
    def points(self):
        """List of variable points."""
        return [self.get_points(dim) for dim in range(self.var_ndim)]

    @property
    def offset(self):
        """List of variable offsets."""
        return [self.get_offset(dim) for dim in range(self.var_ndim)]

    @property
    def center(self):
        """List of variable centers."""
        return [self.get_center(dim) for dim in range(self.var_ndim)]

    @property
    def step(self):
        """List of variable steps."""
        return [self.get_step(dim) for dim in range(self.var_ndim)]

    @property
    def low(self):
        """List of variable minima."""
        return [self.get_low(dim) for dim in range(self.var_ndim)]

    @property
    def high(self):
        """List of variable maxima."""
        return [self.get_high(dim) for dim in range(self.var_ndim)]

    @property
    def bins(self):
        """List of variable bins."""
        return [self.get_bins(dim) for dim in range(self.var_ndim)]

    @property
    def data(self):
        """Data array."""
        return self._data

    @data.setter
    def data(self, val):
        self._data = misc.assume_numpy_array(val)
        diff_ndim = self.ndim - self.var_ndim
        if diff_ndim > 0:
            self.add_dim(diff_ndim)
        elif diff_ndim < 0:
            self.rmv_dim(num=np.abs(diff_ndim))
        self.var_shape = self.shape

    def __getitem__(self, key):
        """Get sliced `ArrayData` object. Supports variable change."""
        if np.isscalar(key) or isinstance(key, slice):
            key = (key,)
        # If single item
        if (len(key) == self.ndim and np.all([np.isscalar(k) for k in key])):
            return self.data[key]
        # Else (slice in at least one dimension)
        obj = self.copy_var()
        kdim = len(key)
        for i, k in enumerate(reversed(key)):
            dim = kdim - i - 1
            # If single entry
            if np.isscalar(k):
                obj.rmv_dim(dim)
            # If slice
            elif isinstance(k, slice):
                _var_mode = obj.var_mode[dim]
                _size = obj.shape[dim]
                # Normalize slice
                _kstart, _kstop, _kstep = 0, _size, 1
                if k.start is not None:
                    _kstart = k.start if k.start >= 0 else k.start % _size
                if k.step is not None:
                    _kstep = k.step
                if k.stop is not None:
                    _kstop = k.stop if k.stop >= 0 else k.stop % _size
                    _kstop = (((_kstop - _kstart) // _kstep - 1) * _kstep
                              + _kstart + 1)
                if _kstop < _kstart:
                    continue
                # Slice variable
                if _var_mode == self.POINTS:
                    _points = obj.get_points(dim)
                    obj.set_dim(dim, points=_points[k])
                elif _var_mode == self.RANGE:
                    _offset, _step = obj.get_offset(dim), obj.get_step(dim)
                    _offset += _kstart * _step
                    _step *= _kstep
                    obj.set_dim(dim, offset=_offset, step=_step)
                elif _var_mode == self.LINSPACE:
                    _low, _step = obj.get_low(dim), obj.get_step(dim)
                    _low += _kstart * _step
                    _high = _low + (_kstop - _kstart - 1) * _kstep * _step
                    obj.set_dim(dim, low=_low, high=_high)
                else:
                    raise ValueError(f"invalid mode: {_var_mode}")
            # Else (error)
            else:
                raise ValueError(f"invalid key type ({k} of type {type(k)})")
        obj.data = self.data[key]
        return obj

    def __setitem__(self, key, val):
        """Set data array item by index."""
        self.data[key] = val

    def __iter__(self):
        """
        To prevent slow iteration with :py:meth:`__getitem__`,
        returns `np.ndarray` iterator.
        """
        return iter(self.data)

    def __str__(self):
        s = f"{str(self.data_quantity)}\n"
        s += "Quantity:\n["
        for i in range(self.ndim):
            if i > 0:
                s += " "
            s += str(self.var_quantity[i])
            if i < self.ndim - 1:
                s += ",\n"
        s += "]\n"
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

    # ++++++++++
    # Conversion
    # ++++++++++

    def copy(self):
        """
        Returns a deep copy of the object.
        """
        return copy.deepcopy(self)

    def copy_var(self):
        """
        Returns a deep copy of all objects except for :py:attr:`data`,
        which is copied by reference.
        """
        obj = self.__class__()
        for attr_name in [
            "data_quantity", "_placeholder_shape",
            "var_quantity", "var_mode",
            "_points",
            "_offset", "_center", "_step",
            "_low", "_high"
        ]:
            setattr(obj, attr_name, copy.deepcopy(getattr(self, attr_name)))
        obj.data = self.data
        return obj

    def cv_index_to_quantity(self, ind, dim):
        """
        Converts a variable array index to the corresponding variable
        physical quantity value.

        Parameters
        ----------
        ind :
            Index to be converted.
        dim : `int`
            Dimension.

        Returns
        -------
        val :
            Physical quantity associated with given quantity.
        """
        return self.get_offset(dim) + self.get_step(dim) * ind

    def cv_quantity_to_index(self, val, dim):
        """
        Converts a variable physical quantity value to the corresponding
        variable array index.

        Parameters
        ----------
        val : numeric
            Physical quantity value to be converted.
        dim : `int`
            Dimension.

        Returns
        -------
        ind : `int`
            Index associated with given quantity.

        Raises
        ------
        IndexError:
            If `dim` is out of bounds.
        """
        if self.var_mode[dim] == self.POINTS:
            ind = np.argmin(np.abs(self.get_points(dim) - val))
        else:
            val = max(val, self.get_offset(dim))
            ind = round((val - self.offset[dim]) / self.scale[dim])
        return ind

    # +++++++++++++
    # Interpolation
    # +++++++++++++

    def __call__(self, var, mode="nearest", extrapolation=False):
        """
        Acts as a function and interpolates for the given `var`.

        Parameters
        ----------
        var : `list(float)` or `list(np.ndarray(float))`
            Requested variables for which the functional value is
            obtained. The variable format must be a list of each
            variable dimension (typically a flattened ij-indexed
            meshgrid). Supports higher dimensional meshgrids, but
            is discouraged because of index ambiguity.
            Shape:
                (data dimension, *) where * can be any scalar
                or array.
        mode : `str`
            `"nearest"`: Value of nearest neighbour.
            `"linear"`: Linear interpolation.
        extrapolation : `bool` or `float`
            `True`: Performs nearest/linear extrapolation.
            `False`: Raises ValueError if extrapolated value is
                   requested.
            `float`: Used as extrapolation fill value.

        Returns
        -------
        func_val : `np.ndarray(float)`
            Functional values of given variables.
            Shape: `var.shape[1:]`.

        Raises
        ------
        ValueError
            See parameter `extrapolation`.

        Notes
        -----
        * See also scipy.interpolate.interpn:
          <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn>`
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

    def supersample(self, rep):
        """
        Gets a supersampled `ArrayData` object.

        Parameters
        ----------
        rep : `int` or `Iter[int]`
            Supersampling repetitions (number of new indices per old index).
            If iterable, each value corresponds to the respective dimension.

        Returns
        -------
        ad : `ArrayData`
            Supersampled array data.
        """
        if np.isscalar(rep):
            rep = self.ndim * (rep,)
        if rep != self.ndim:
            raise ValueError("invalid dimensions")
        ad = self.copy()
        for i in range(self.ndim):
            _rep = int(round(rep[i]))
            if _rep > 1:
                ad.data = ad.data.repeat(_rep, axis=i)
                _new_step = ad.get_step(i) / _rep
                if ad.var_mode[i] == self.RANGE:
                    ad._step[i] = _new_step
                    if ad._offset[i] is not None:
                        ad._offset[i] -= _new_step * (_rep - 1) / 2
                elif ad.var_mode[i] == self.LINSPACE:
                    ad._low[i] -= _new_step * (_rep - 1) / 2
                    ad._high[i] += _new_step * (_rep - 1) / 2
                elif ad.var_mode[i] == self.POINTS:
                    raise NotImplementedError("POINTS mode not implemented")
        return ad

    def pad(self, shape, val=0):
        """
        Gets a resized `ArrayData` with the old data centered and padded.

        Can also be used to obtain a centered slice.

        Parameters
        ----------
        shape : `tuple(int)` or `float`
            Padded shape.
            If `float`, scales all dimensions by the factor `shape`.
        val : `Any`
            Padding value.

        Returns
        -------
        ad : `ArrayData`
            Padded array data with `shape`.
        """
        # Parse arguments
        if shape is None:
            return self.copy()
        if np.isscalar(shape):
            shape = tuple(np.round(shape * np.array(self.shape)).astype(int))
        if len(shape) != self.ndim:
            raise ValueError("invalid dimensions")
        # Setup array data
        ad = self.copy_var()
        ad.data = np.full(shape, val, dtype=self.data.dtype)
        # Find lower indices
        shape_half_diff = [
            (shape[i] - self.shape[i]) // 2 for i in range(self.ndim)
        ]
        # Find slices for old and new array
        slice_old, slice_new = [], []
        for i in range(self.ndim):
            # New shape is larger
            if shape_half_diff[i] >= 0:
                slice_old.append(slice(None))
                slice_new.append(slice(
                    shape_half_diff[i], shape_half_diff[i] + self.shape[i]
                ))
            # Old shape is larger
            else:
                slice_old.append(slice(
                    -shape_half_diff[i], -shape_half_diff[i] + ad.shape[i]
                ))
                slice_new.append(slice(None))
        slice_old, slice_new = tuple(slice_old), tuple(slice_new)
        ad.data[slice_new] = self.data[slice_old]
        # Change variable data
        for i in range(self.ndim):
            if ad.var_mode[i] == self.RANGE:
                if ad._offset[i] is not None:
                    ad._offset[i] -= shape_half_diff[i] * ad.get_step(i)
            elif ad.var_mode[i] == self.LINSPACE:
                ad._low[i] -= shape_half_diff[i] * ad.get_step(i)
                ad._high[i] += shape_half_diff[i] * ad.get_step(i)
            elif ad.var_mode[i] == self.POINTS:
                raise NotImplementedError("POINTS mode not implemented")
        return ad

    # ++++++++++++++++++++++
    # Combination operations
    # ++++++++++++++++++++++

    def cmp_quantity(self, other):
        """
        Compares whether the quantities of two ArrayData objects are identical.
        """
        for dim, _vq in enumerate(self.var_quantity):
            if not _vq != other.var_quantity[dim]:
                return False
        if self.data_quantity != other.data_quantity:
            return False
        return True

    def cmp_shape(self, other):
        """
        Compares whether the shapes of two ArrayData objects are identical.
        """
        return self.data.shape == other.data.shape

    def cmp_var(self, other):
        """
        Compares whether the variables of two ArrayData objects are identical.
        """
        return np.all([
            np.allclose(self.get_points(dim), other.get_points(dim))
            for dim in range(self.ndim)
        ])

    def get_common_obj(self, other, op, in_place=False, rev=False, raw=False):
        """
        Creates the common result after operation.

        Parameters
        ----------
        op : `callable`
            Function signature:
            `op(numpy.ndarray, numpy.ndarray) -> numpy.ndarray`.
        in_place : `bool`
            Flag whether to store result in same instance.
        rev : `bool`
            Flag whether to reverse operand order, i.e. to call
            `op(other, self)` instead of `op(self, other)`.
        raw : `bool`
            Flag whether to return only the data, i.e. without
            the `ArrayData` container.

        Returns
        -------
        obj : `ArrayData` or `numpy.ndarray`
            Processed object.
        """
        obj = self if in_place else copy.deepcopy(self)
        # Non-homogeneous operation
        if not isinstance(other, ArrayData):
            if rev:
                obj.data = op(other, self.data)
            else:
                obj.data = op(self.data, other)
            return obj
        # Perform operation on data array
        if rev:
            obj.data = op(other.data, obj.data)
        else:
            obj.data = op(obj.data, other.data)
        # Return type
        if raw:
            return obj.data
        else:
            return obj

    def get_copy_obj(self, op, raw=False):
        """
        Creates a copy of self object and applies given operation on it.

        Parameters
        ----------
        op : `callable`
            Function signature:
            `op(numpy.ndarray) -> numpy.ndarray`
        raw : `bool`
            Flag whether to return only the data, i.e. without
            the `ArrayData` container.

        Returns
        -------
        obj : `ArrayData`
            Processed object.
        """
        obj = copy.deepcopy(self)
        obj.data = op(obj.data)
        if raw:
            return obj.data
        else:
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

    def __and__(self, other):
        return self.get_common_obj(other, np.bitwise_and)

    def __or__(self, other):
        return self.get_common_obj(other, np.bitwise_or)

    def __xor__(self, other):
        return self.get_common_obj(other, np.bitwise_xor)

    # ++++ Reflected arithmetics ++

    def __radd__(self, other):
        return self.get_common_obj(other, np.add, rev=True)

    def __rsub__(self, other):
        return self.get_common_obj(other, np.subtract, rev=True)

    def __rmul__(self, other):
        return self.get_common_obj(other, np.multiply, rev=True)

    def __rtruediv__(self, other):
        return self.get_common_obj(other, np.true_divide, rev=True)

    def __rfloordiv__(self, other):
        return self.get_common_obj(other, np.floor_divide, rev=True)

    def __rmod__(self, other):
        return self.get_common_obj(other, np.mod, rev=True)

    def __rpow__(self, other):
        return self.get_common_obj(other, np.power, rev=True)

    def __rand__(self, other):
        return self.get_common_obj(other, np.bitwise_and, rev=True)

    def __ror__(self, other):
        return self.get_common_obj(other, np.bitwise_or, rev=True)

    def __rxor__(self, other):
        return self.get_common_obj(other, np.bitwise_xor, rev=True)

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

    # ++++ Comparisons +++++++++++

    def __lt__(self, other):
        return np.all(self.get_common_obj(other, np.less, raw=True))

    def __le__(self, other):
        return np.all(self.get_common_obj(other, np.less_equal, raw=True))

    def __eq__(self, other):
        try:
            return np.all(self.get_common_obj(other, np.equal, raw=True))
        except ValueError:
            return False

    def __ne__(self, other):
        try:
            return np.all(self.get_common_obj(other, np.not_equal, raw=True))
        except ValueError:
            return True

    def __ge__(self, other):
        return np.all(self.get_common_obj(other, np.greater_equal, raw=True))

    def __gt__(self, other):
        return np.all(self.get_common_obj(other, np.greater, raw=True))

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

    # ++++ Numpy universal functions

    def __array_ufunc__(self, ufunc, method, i, *inputs, **kwargs):
        # Convert ArrayData inputs into np.ndarray inputs
        if isinstance(i, ArrayData):
            i = i.data
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
            res = self.data.__array_ufunc__(
                ufunc, method, i, *inputs, **kwargs
            )
            if np.isscalar(res):
                obj = res
            else:
                obj = copy.deepcopy(self)
                obj.data = res
        # Return ArrayData object with data as calculated by the ufunc
        return obj


###############################################################################
# SeriesData
###############################################################################


class SeriesData(object):

    def __init__(self):
        self._data = np.empty((0, 0))
        self.quantity = []

    def set_quantity(self, dim, **kwargs):
        """
        Sets the quantity object.


        Parameters
        ----------
        dim : `int`
            Variable dimension to be set.
        **kwargs
            See :py:func:`assume_quantity`.
        """
        _quantity = assume_quantity(**kwargs)
        self.quantity[dim] = _quantity

    def add_dim(self, *args):
        """
        Appends dimension(s) to the object.

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
        # Handle multiple dimensions
        if len(args) == 1 and isinstance(args[0], int):
            args = args[0] * [{}]
        if not isinstance(args[0], dict):
            for arg in args:
                self.add_dim(arg)
            return
        # Handle single dimension
        arg = args[0]
        if not isinstance(arg, dict):
            raise ValueError("invalid argument: {:s}".format(str(arg)))
        # Add dimensions
        self.quantity.append(None)
        # Set dimensions
        self.set_quantity(-1, **arg)

    def rmv_dim(self, *dims, num=None):
        """
        Removes dimension(s) from the object.

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
            del self.quantity[dim]

    # ++++++++++
    # Properties
    # ++++++++++

    @property
    def ndim(self):
        return len(self._data)

    @property
    def total_ndim(self):
        return self.ndim

    @property
    def shape(self):
        return self._data.shape

    def __len__(self):
        return self._data.shape[-1]

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

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, val):
        self.data[key] = val

    def __str__(self):
        s = "Quantity:\n["
        for i in range(self.ndim):
            if i > 0:
                s += " "
            s += str(self.quantity[i])
            if i < self.ndim - 1:
                s += ",\n"
        s += "]\n"
        s += "Data:\n"
        s += "{:s}".format(str(self.data))
        return s


###############################################################################
# Miscellaneous
###############################################################################


def assume_quantity(**kwargs):
    """
    Generates a quantity object from keyword arguments.

    Parameters
    ----------
    quantity : `types.Quantity`
        Directly specified quantity. Takes precedence
        over other methods.
    name, symbol, unit : `str`
        Parameters used to construct `types.Quantity` instance.

    Returns
    -------
    quantity : `types.Quantity`
        Constructed object.
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
