import numpy as np

from libics.data import types
from libics.file import hdf
from libics.util import InheritMap


###############################################################################


@InheritMap(map_key=("libics", "SeriesData"))
class SeriesData(hdf.HDFBase):

    def __init__(self, dim=0):
        super().__init__(pkg_name="libics", cls_name="SeriesData")
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

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        if val.shape[0] == len(self.quantity):
            self._data = val
        else:
            raise ValueError("invalid data shape")

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

        Example
        -------
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
