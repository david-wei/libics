# System Imports
import copy
import numpy as np
import operator

# Package Imports
from libics.cfg import err as ERR
from libics.file import hdf
from libics.util import misc

# Subpackage Imports
from libics.data import types


###############################################################################


class ArrayScale(hdf.HDFBase):

    """
    Stores metadata for `ArrayData`.

    Provides linear scaling (offset, scale) for discrete arrays and physical
    quantities (name, unit) for each dimension.

    Parameters
    ----------
    dim : int, optional
        Dimension of associated `ArrayData` + 1.

    Notes
    -----
    The stored dimensions must be one higher than the stored data array to
    accomodate the dimension of the function value.

    Attributes
    ----------
    INDEX, QUANTITY:
        Mode determining how an array item is addressed, i.e.
        whether the value given to the getter function is
        interpreted as index or as physical quantity.
    """

    INDEX = 0
    QUANTITY = 1

    def __init__(self, dim=0):
        super().__init__(pkg_name="libics", cls_name="ArrayScale")
        self.init(dim)

    def init(self, dim=0):
        dim = round(dim)
        ERR.assertion(ERR.INVAL_POS, dim >= 0)
        self.offset = dim * [0.0]
        self.scale = dim * [1.0]
        self.max = dim * [None]
        self.quantity = dim * [types.Quantity()]
        self.get_index_func = self.get_index_by_index
        self.set_index_mode(ArrayScale.INDEX)

    def set_index_mode(self, index_mode):
        """
        Sets the mode how the `__getitem__` operator (`[]`) is interpreted.

        Parameters
        ----------
        index_mode : ArrayScale.INDEX, ArrayScale.QUANTITY
            INDEX: Interpretation as index.
            QUANTITY: Interpretation as physical quantity.
        """
        if index_mode == ArrayScale.INDEX:
            self.get_index_func = self.get_index_by_index
        elif index_mode == ArrayScale.QUANTITY:
            self.get_index_func = self.get_index_by_quantity

    def set_max(self, ar):
        """
        Determines the quantity maxima from the stored scaling and a given
        array.

        Parameters
        ----------
        ar : numpy.ndarray or tuple
            Array (or its shape) from which the maximum is determined.
        """
        if type(ar) == np.ndarray:
            ar = ar.shape
        self.max = [self.cv_index_to_quantity(ind, dim)
                    for dim, ind in enumerate(ar)]

    def get_index_by_index(self, val, **kwargs):
        """
        Identity function.
        """
        return val

    def get_index_by_quantity(self, val, dim=None):
        """
        Gets an index combination.

        Parameters
        ----------
        val : scalar, slice, tuple, list, numpy.ndarray
            Any argument of numpy fancy indexing.
        dim : int or None
            None: Dimension starting from 0.
            int: dimension for scalar.
        """
        ind = None
        if type(val) == tuple or type(val) == list or type(val) == np.ndarray:
            ind = tuple([self.get_index_by_quantity(v, dim=d)
                         for d, v in enumerate(val)])
        elif type(val) == slice:
            ind = slice(
                self.get_index_by_quantity(val.start, dim=dim),
                self.get_index_by_quantity(val.stop + self.scale[dim],
                                           dim=dim),
                self.get_index_by_quantity(val.step, dim=dim)
            )
        elif val is not None:
            ind = self.cv_quantity_to_index(val, dim)
        return ind

    def get_index(self, val):
        """
        Gets the index associated with the given value being interpreted
        according to the index mode.

        Parameters
        ----------
        val :
            Physical quantity or index.

        Returns
        -------
        ind : int
            Requested index.
        """
        return self.get_index_func(val)

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
        return self.offset[dim] + self.scale[dim] * ind

    def cv_quantity_to_index(self, val, dim, round_index=True):
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
        val = max(val, self.offset[dim])
        if self.max[dim] is not None:
            val = min(val, self.max[dim])
        ind = (val - self.offset[dim]) / self.scale[dim]
        if round_index:
            ind = round(ind)
        return ind

    def chk_attr(self):
        """
        Checks attribute consistency (dimensionality).

        Returns
        -------
        ret : bool
            Whether attributes are consistent.
        """
        ret = True
        try:
            ret = (len(self.offset) == len(self.scale)
                   == len(self.quantity) == len(self.max))
        except(TypeError):
            ret = False
        return ret

    def get_dim(self):
        """
        Gets the dimension of stored attributes.

        Raises
        ------
        AttributeError:
            If dimensionality is inconsistent.
        """
        if not self.chk_attr():
            raise AttributeError
        else:
            return len(self.offset)

    def add_dim(self,
                offset=0.0, scale=1.0, max=None,
                quantity=None, name="N/A", symbol=None, unit=None):
        """
        Appends a dimension to the object.

        Parameters
        ----------
        offset : float or int
            Scaling offset.
        scale : float or int
            Linear scaling.
        max : float or int or None
            Maximum quantity.
        quantity : types.Quantity or None
            Quantity. If specified, overwrites name and unit.
        name : str
            Quantity name.
        symbol : str or None
            Quantity symbol.
        unit : str or None
            Quantity unit.
        """
        self.offset.append(offset)
        self.scale.append(scale)
        self.max.append(max)
        if quantity is not None:
            self.quantity.append(quantity)
        else:
            self.quantity.append(
                types.Quantity(name=name, symbol=symbol, unit=unit)
            )

    def __str__(self):
        return (
            "Offset: " + str(self.offset) + "\n"
            + "Scale: " + str(self.scale) + "\n"
            + "Max: " + str(self.max) + "\n"
            + "Quantity: " + str(self.quantity)
        )

    def __repr__(self):
        return str(self)


###############################################################################


class ArrayData(hdf.HDFBase):

    """
    Stores a multidimensional array and its scaling information (linear
    scaling and physical quantity).

    Attributes
    ----------
    ROUND, UNIFORM, NORM1, NORM2, NORMINF:
        Mode determining how a float index item is retrieved,
        i.e. whether and how the index value is interpolated.
    """

    ROUND = 0
    UNIFORM = 1
    NORM1 = 2
    NORM2 = 3
    NORMINF = 4

    def __init__(self):
        super().__init__(pkg_name="libics", cls_name="ArrayData")
        self.init()

    def init(self):
        self.data = None
        self.scale = ArrayScale()
        self.get_float_item_func = self.get_float_item_by_round

    # ++++ Scale +++++++++++++++++++++++++

    def set_index_mode(self, index_mode):
        """
        Sets the mode how the `__getitem__` operator (`[]`) is interpreted.

        Parameters
        ----------
        index_mode : ArrayScale.CONST
            INDEX: Interpretation as index.
            QUANTITY: Interpretation as physical quantity.

        Notes
        -----
        If an invalid parameter is given, the index mode remains unchanged.
        """
        return self.scale.set_index_mode(index_mode)

    def set_max(self):
        """
        Determines the quantity maxima from the stored scale and data.
        """
        self.scale.set_max(self.data)

    # ++++ Float index +++++++++++++++++++

    def set_float_index_mode(self, float_index_mode):
        """
        Sets the mode how a fractional index (get_float_item) is interpreted.

        Parameters
        ----------
        float_index_mode : ArrayData.CONST
            ROUND: Rounds to closest integer.
            UNIFORM: Averages uniformly with surrounding points.
            NORM1: Averages with 1-norm as weight.
            NORM2: Averages with Euclidean distance as weight.
            NORMINF: Averages with inf-norm as weight.

        Notes
        -----
        If an invalid parameter is given, the index mode remains unchanged.
        """
        if float_index_mode == ArrayData.ROUND:
            self.get_float_item_func = self.get_float_item_by_round
        elif float_index_mode == ArrayData.UNIFORM:
            self.get_float_item_func = self.get_float_item_by_uniform
        elif float_index_mode == ArrayData.NORM1:
            self.float_item_norm = 1
            self.get_float_item_func = self.get_float_item_by_norm
        elif float_index_mode == ArrayData.NORM2:
            self.float_item_norm = 2
            self.get_float_item_func = self.get_float_item_by_norm
        elif float_index_mode == ArrayData.NORMINF:
            self.float_item_norm = np.inf
            self.get_float_item_func = self.get_float_item_by_norm

    def get_float_item(self, ind):
        return self.get_float_item_func(tuple(ind))

    def get_float_item_by_round(self, ind):
        ind = [round(x) for x in ind]
        if len(ind) == 1:
            ind = ind[0]
        return self.data[ind]

    @staticmethod
    def _cv_float_index_to_int_index(ind):
        inds = []
        for x in ind:
            xx = int(x)
            if np.isclose(x, xx):
                inds.append((xx, ))
            else:
                inds.append((xx, xx + 1))
        return inds

    def get_float_item_by_uniform(self, ind):
        inds = ArrayData._cv_float_index_to_int_index(ind)
        items = [self.data[ind] for ind in misc.get_combinations(inds)]
        return np.mean(items)

    def get_float_item_by_norm(self, ind):
        inds = ArrayData._cv_float_index_to_int_index(ind)
        inds = misc.get_combinations(inds)
        weights = []
        ind = np.array(ind)
        for i in inds:
            weights.append(np.linalg.norm(ind - np.array(i)))
        weights = np.array(weights) / np.sum(weights)
        items = np.array([self.data[i] for i in inds])
        return items * weights

    # ++++ Conversion +++++++++++++++++++++

    def cv_unit(self, dim, new_unit, op, val):
        """
        Converts the unit of a given dimension.

        Parameters
        ----------
        dim : int
            Dimension of conversion.
        new_unit : str
            New unit string.
        op : str or function
            Performed operation as function or encoded as string.
            str:
                Supported string encoded operations include:
                "+", "-", "*", "/".
                Passed val parameter is used as second
                operand.
            function:
                Call signature must be:
                function(stored_value, val).
        val : int or float
            Value with which the operation is performed.
        """
        if not callable(op):
            try:
                op = misc.operator_mapping[op]
            except(KeyError):
                return
        self.scale.quantity[dim].unit = new_unit
        if dim < self.scale.get_dim() - 1:
            self.scale.offset[dim] = op(self.scale.offset[dim], val)
            self.scale.scale[dim] = op(self.scale.scale[dim], val)
            if self.scale.max[dim] is not None:
                self.scale.max[dim] = op(self.scale.max[dim], val)
        else:
            self.data = op(self.data, val)

    # ++++ Self operations ++++++++++++++++

    def __getitem__(self, key):
        return self.data[self.scale.get_index(key)]

    def __setitem__(self, key, val):
        self[key] = val

    def __str__(self):
        return str(self.scale) + "\nData:\n" + str(self.data)

    def __repr__(self):
        return str(self)

    def chk_attr(self):
        """
        Checks the object's attribute consistency (data and scale dimension).

        Returns
        -------
        ret : bool
            Whether attributes are consistent.
        """
        ret = True
        try:
            ret = (len(self.data.shape) == self.scale.get_dim() - 1)
        except(TypeError, AttributeError):
            ret = False
        return ret

    # ++++ Combination operations ++++++++

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

    def get_common_obj(self, other, op):
        """
        Creates a copy of self.

        The data space is chosen to be common to both ArrayData objects.
        The given operation is performed to combine both.

        Parameters
        ----------
        op : callable
            Function signature:
            op(numpy.ndarray, numpy.ndarray) -> numpy.ndarray.

        Returns
        -------
        obj : ArrayData
            Processed object.

        Raises
        ------
        ValueError:
            If the two ArrayData objects cannot be combined.
        """
        if not self.cmp_necessary(other):
            raise ValueError("Invalid necessary values")
        obj = ArrayData()
        obj.scale = copy.deepcopy(self.scale)
        # FIXME: including/excluding max index ambiguity
        (
            obj.scale.offset, obj.scale.max,
            si_offset, si_max,
            oi_offset, oi_max
         ) = self.get_common_rect(other)
        obj.data = self.data[
            [slice(si_offset[i], si_max[i]) for i in range(len(si_offset))]
        ]
        if self.chk_commensurable():
            obj.data = op(
                obj.data,
                other.data[
                    [slice(oi_offset[i], oi_max[i])
                     for i in range(len(oi_offset))]
                ]
            )
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

    def __add__(self, other):
        return self.get_common_obj(other, operator.__add__)

    def __sub__(self, other):
        return self.get_common_obj(other, operator.__sub__)

    def __mul__(self, other):
        return self.get_common_obj(other, operator.__mul__)

    def __truediv__(self, other):
        return self.get_common_obj(other, operator.__truediv__)

    def __floordiv__(self, other):
        return self.get_common_obj(other, operator.__floordiv__)

    def __and__(self, other):
        return self.get_common_obj(other, operator.__and__)

    def __xor__(self, other):
        return self.get_common_obj(other, operator.__xor__)

    def __or__(self, other):
        return self.get_common_obj(other, operator.__or__)

    def __pow__(self, other):
        return self.get_common_obj(other, operator.__pow__)

    def __lt__(self, other):
        return self.get_common_obj(other, operator.__lt__).data

    def __le__(self, other):
        return self.get_common_obj(other, operator.__le__).data

    def __eq__(self, other):
        return self.get_common_obj(other, operator.__eq__).data

    def __ne__(self, other):
        return self.get_common_obj(other, operator.__ne__).data

    def __ge__(self, other):
        return self.get_common_obj(other, operator.__ge__).data

    def __gt__(self, other):
        return self.get_common_obj(other, operator.__gt__).data

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

    def __invert__(self):
        return self.get_copy_obj(operator.__invert__)

    def __neg__(self):
        return self.get_copy_obj(operator.__neg__)


###############################################################################


if __name__ == "__main__":

    # Create test data (image)
    data = np.arange(100, dtype="float64").reshape((10, 10))
    pq_data = types.Quantity(name="intensity", symbol="I", unit="mW/cm2")
    pixel_size = 6.45
    pixel_offset = 0.0
    pq_x_pos = types.Quantity(name="position", symbol="x", unit="µm")
    pq_y_pos = types.Quantity(name="position", symbol="y", unit="µm")

    # Load into data structure
    ardata = ArrayData()
    ardata.data = data
    ardata.scale.init()
    ardata.scale.add_dim(pixel_offset, pixel_size, quantity=pq_x_pos)
    ardata.scale.add_dim(pixel_offset, pixel_size, quantity=pq_y_pos)
    ardata.scale.add_dim(pixel_offset, pixel_size, quantity=pq_data)
    print("------\nardata\n------\n" + str(ardata) + "\n")

    # Slice data in QUANTITY mode
    ardata.set_index_mode(ArrayScale.QUANTITY)
    print("-------------\nsliced ardata\n-------------\n"
          + str(ardata[6.4:13, :20]))
