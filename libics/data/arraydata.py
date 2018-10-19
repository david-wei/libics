# System Imports
import numpy as np

# Package Imports
from libics.cfg import err as ERR
from libics.file import hdf

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
        self.get_index_func = lambda x: x
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
            self.get_index_func = lambda x: x
        elif index_mode == ArrayScale.QUANTITY:
            self.get_index_func = self.get_by_quantity

    def get_by_quantity(self, val, dim=None):
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
            ind = tuple([self.get_by_quantity(v, dim=d)
                         for d, v in enumerate(val)])
        elif type(val) == slice:
            ind = slice(
                self.get_by_quantity(val.start, dim=dim),
                self.get_by_quantity(val.stop + self.scale[dim], dim=dim),
                self.get_by_quantity(val.step, dim=dim)
            )
        elif val is not None:
            ind = self.quantity_to_index(val, dim)
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

    def index_to_quantity(self, ind, dim):
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

    def quantity_to_index(self, val, dim):
        """
        Converts a physical quantity value to an array index.

        Parameters
        ----------
        val :
            Physical quantity value to be converted.
        dim : int
            Dimension.

        Returns
        -------
        ind : int
            Index associated with given quantity.
        """
        val = max(val, self.offset[dim])
        if self.max[dim] is not None:
            val = min(val, self.max[dim])
        return round((val - self.offset[dim]) / self.scale[dim])

    def check_attr_cons(self):
        """
        Checks attribute consistency (dimensionality).

        Returns
        -------
        ret : bool
            True: if consistent,
            False: if otherwise.
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
        if not self.check_attr_cons():
            raise AttributeError
        else:
            return len(self.offset)

    def add_dim(self,
                offset=0.0, scale=1.0, max=None,
                quantity=None, name="N/A", unit=None):
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
        unit : str or None
            Quantity unit.
        """
        self.offset.append(offset)
        self.scale.append(scale)
        self.max.append(max)
        if quantity is not None:
            self.quantity.append(quantity)
        else:
            self.quantity.append(types.Quantity(name=name, unit=unit))

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
    """

    def __init__(self):
        super().__init__(pkg_name="libics", cls_name="ArrayData")
        self.init()

    def init(self):
        self.data = None
        self.scale = ArrayScale()

    def set_index_mode(self, index_mode):
        """
        Sets the mode how the `__getitem__` operator (`[]`) is interpreted.

        Parameters
        ----------
        index_mode : ArrayScale.INDEX, ArrayScale.QUANTITY
            INDEX: Interpretation as index.
            QUANTITY: Interpretation as physical quantity.

        Notes
        -----
        If an invalid parameter is given, the index mode remains unchanged.
        """
        return self.scale.set_index_mode(index_mode)

    def __getitem__(self, key):
        return self.data[self.scale.get_index(key)]

    def __setitem__(self, key, val):
        self[key] = val

    def check_attr_cons(self):
        """
        Checks the object's attribute consistency (data and scale dimension).

        Returns
        -------
        ret : bool
            True: if attributes are consistent,
            False: otherwise.
        """
        ret = True
        try:
            ret = (len(self.data.shape) == self.get_dim() - 1)
        except(TypeError, AttributeError):
            ret = False
        return ret

    def __str__(self):
        return str(self.scale) + "\nData:\n" + str(self.data)

    def __repr__(self):
        return str(self)


###############################################################################


if __name__ == "__main__":

    # Create test data (image)
    data = np.arange(100, dtype="float64").reshape((10, 10))
    pq_data = types.Quantity(name="intensity", unit="mW/cm2")
    pixel_size = 6.45
    pixel_offset = 0.0
    pq_x_pos = types.Quantity(name="x position", unit="µm")
    pq_y_pos = types.Quantity(name="y position", unit="µm")

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
