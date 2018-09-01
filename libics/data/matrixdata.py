# System Imports
import numpy as np

# Package Imports
from libics.cfg import err as ERR
from libics.util import serialization as ser

# Subpackage Imports
from libics.data import types


###############################################################################


class MatrixRect(ser.DictSerialization, object):

    """
    Stores matrix bounds (as index and as physical quantity) of a numpy matrix.
    """

    def __init__(self):
        # Index boundary (upper bound not included)
        self.ind_bound_x = [0, 0]   # Matrix index bounds on x axis
        self.ind_bound_y = [0, 0]   # Matrix index bounds on y axis
        # Physical value boundary (upper bound included)
        self.pquant_bound_x = [0.0, 0.0]    # Phys. quantity bounds on x axis
        self.pquant_bound_y = [0.0, 0.0]    # Phys. quantity bounds on y axis

    #   #### Setter ####################

    def set_rect_by_pixels(self,
                           matrix,
                           pq_pxsize_x=1.0, pq_pxsize_y=1.0,
                           pq_offset_x=0.0, pq_offset_y=0.0):
        """
        Sets the boundaries by defining pixel size and matrix offset in units
        of the physical quantity.

        Parameters
        ----------
        matrix : `numpy.ndarray(2)`
            Matrix from which the index boundaries are determined.
        pq_pxsize_x, pq_pxsize_y : `float`, optional
            Pixel sizes in x and y directions in units of physical
            quantity.
        pq_offset_x, pq_offset_y : `float`, optional
            Matrix offset in x and y directions in units of physical
            quantity.
        """
        # Set index boundary
        self.ind_bound_x = [0, matrix.shape[0]]
        self.ind_bound_y = [0, matrix.shape[1]]
        # Calculate physical value boundary
        self.pquant_bound_x = [
            pq_offset_x,
            pq_offset_x
            + pq_pxsize_x * (self.ind_bound_x[1] - 1 - self.ind_bound_x[0])
        ]
        self.pquant_bound_y = [
            pq_offset_y,
            pq_offset_y
            + pq_pxsize_y * (self.ind_bound_y[1] - 1 - self.ind_bound_y[0])
        ]

    def set_rect_by_bounds(self,
                           matrix,
                           pq_bound_x=None, pq_bound_y=None):
        """
        Sets the boundaries by defining the physical quantity bounds.

        Parameters
        ----------
        matrix : `numpy.ndarray(2)`
            Matrix from which the index boundaries are determined.
        pq_bound_x, pq_bound_y : `[float, float]` or `None`
            [Minimum, maximum] of physical quantity bounds in x and y
            directions.
            `None` uses index bounds as physical bounds.
        """
        # Set index boundary
        self.ind_bound_x = [0, matrix.shape[0]]
        self.ind_bound_y = [0, matrix.shape[1]]
        # Calculate physical value boundary
        if pq_bound_x is None:
            self.pquant_bound_x = self.ind_bound_x
        else:
            self.pquant_bound_x = pq_bound_x
        if pq_bound_y is None:
            self.pquant_bound_y = self.ind_bound_y
        else:
            self.pquant_bound_y = pq_bound_y

    #   #### Calculated Getter #########

    def get_pxsize(self):
        """
        Calculates and returns the pixel size in physical units.

        Returns
        -------
        pxsize : `[float, float]`
            Pixel size in [x, y] direction in physical units.
        """
        x_ind_range = self.ind_bound_x[1] - 1 - self.ind_bound_x[0]
        y_ind_range = self.ind_bound_y[1] - 1 - self.ind_bound_y[0]
        x_pq_pxsize = (
            0.0 if x_ind_range == 0.0 else
            (self.pquant_bound_x[1] - self.pquant_bound_x[0]) / x_ind_range)
        y_pq_pxsize = (
            0.0 if y_ind_range == 0.0 else
            (self.pquant_bound_y[1] - self.pquant_bound_y[0]) / y_ind_range)
        return [x_pq_pxsize, y_pq_pxsize]

    def get_pxcount(self):
        """
        Gets the pixel count.

        Returns
        -------
        x_pxcount, y_pxcount : int
            Pixel count on x, y axis.
        """
        return (self.ind_bound_x[1] - self.ind_bound_x[0],
                self.ind_bound_y[1] - self.ind_bound_y[0])

    def get_pquant_by_index(self,
                            ind_x, ind_y):
        """
        Calculates and returns the given index coordinate in physical units.

        Parameters
        ----------
        ind_x, ind_y : `int`
            x, y index coordinates.

        Returns
        -------
        pquant_coord : `[float, float]`
            [x, y] physical quantity coordinates.
        """
        x_ind_rel = ind_x - self.ind_bound_x[0]
        y_ind_rel = ind_y - self.ind_bound_y[0]
        x_pq_pxsize, y_pq_pxsize = self.get_pxsize()
        x_pq = self.pquant_bound_x[0] + x_pq_pxsize * x_ind_rel
        y_pq = self.pquant_bound_y[0] + y_pq_pxsize * y_ind_rel
        return [x_pq, y_pq]

    def get_index_by_pquant(self,
                            pq_x=None, pq_y=None):
        """
        Calculates and returns the given index coordinate in physical units.

        Parameters
        ----------
        pq_x, pq_y : `float` or `None`
            x, y index coordinates. None uses index 0.

        Returns
        -------
        ind_coord: `[int, int]`
            [x, y] index coordinates.
        """
        x_ind_pxsize, y_ind_pxsize = map(lambda x: 1 / x, self.get_pxsize())
        x_ind = 0
        y_ind = 0
        if pq_x is not None:
            x_pq_rel = pq_x - self.pquant_bound_x[0]
            x_ind = int(round(self.ind_bound_x[0] + x_ind_pxsize * x_pq_rel))
        if pq_y is not None:
            y_pq_rel = pq_y - self.pquant_bound_y[0]
            y_ind = int(round(self.ind_bound_y[0] + y_ind_pxsize * y_pq_rel))
        return [x_ind, y_ind]

    def get_pquant_coordinates(self, axis=0):
        """
        Gets the coordinate array in units of pquant.

        Parameters
        ----------
        axis : 0 or 1
            Coordinate axis (x: 0, y: 1).

        Returns
        -------
        pq_array : numpy.ndarray
            1D array with same length as data describing the pquant
            coordinate position.
        """
        start = (self.pquant_bound_x[0] if axis == 0
                 else self.pquant_bound_y[0])
        stop = (self.pquant_bound_x[1] if axis == 0
                else self.pquant_bound_y[1])
        num = (self.ind_bound_x[1] - self.ind_bound_x[0] if axis == 0
               else self.ind_bound_y[1] - self.ind_bound_y[0])
        pq_array = np.linspace(start, stop, num=num, endpoint=True)
        return pq_array

    def update_rect(self,
                    matrix, scaling="const_pxsize"):
        """
        Updates all bounds according to given matrix shape. Retains the
        previous offset and scales according to `scaling` flag.

        Parameters
        ----------
        matrix : `numpy.ndarray(2)`
            Matrix from which the index boundaries are determined.
        scaling : `"const_pxsize"` or `"const_pqbound"`, optional
            Updates physical unit boundaries assuming constant pixel
            size or constant boundary (default).
        """
        x_pq_pxsize, y_pq_pxsize = None, None
        x_ind_size, y_ind_size = self.ind_bound_x[1], self.ind_bound_y[1]
        if scaling == "const_pxsize":
            x_pq_pxsize, y_pq_pxsize = self.get_pxsize()
        self.ind_bound_x = [0, matrix.shape[0]]
        self.ind_bound_y = [0, matrix.shape[1]]
        if scaling == "const_pxsize":
            x_ind_change = self.ind_bound_x[1] - x_ind_size
            y_ind_change = self.ind_bound_y[1] - y_ind_size
            self.pquant_bound_x[1] += x_ind_change * x_pq_pxsize
            self.pquant_bound_y[1] += y_ind_change * y_pq_pxsize


###############################################################################


class MatrixData(ser.DictSerialization, object):

    """
    Stores a 2D array and its positional (index) data with information
    describing the respective physical quantity and unit.
    """

    def __init__(self):
        super().__init__()
        # Matrix (functional) values
        self.val_data = np.zeros((0, 0))   # Values data (2D array)
        self.val_pquant = None             # Values physical quantity
        # Positional variable (indices)
        self.var_rect = MatrixRect()    # Matrix variable rectangular boundary
        self.var_x_pquant = None        # Variable physical quantity on x axis
        self.var_y_pquant = None        # Variable physical quantity on y axis
        # DictSerialization init
        self.add_obj_ser_func(
            {
                "var_rect": None,
                "val_data": [ser.numpy_ndarray_to_json, self.get_val_data]
            },
            {
                "var_rect": None,
                "val_data": [ser.json_to_numpy_ndarray]
            }
        )

    #   #### Getter ####################

    def get_val_data(self):
        """Gets `val_data` attribute."""
        return self.val_data

    def get_pquants(self):
        """Gets `[var_x_pquant, var_y_pquant, val_pquant]`."""
        return [self.var_x_pquant, self.var_y_pquant, self.val_pquant]

    def __getitem__(self, key):
        """Gets `val_data[key]` by index."""
        return self.val_data[key]

    def get_by_pquant(self, *key):
        """
        Gets `val_data[index(key)]` by pquant.

        Parameters
        ----------
        key : (pq_x) == (pq_x, None)
            Gets the data `x`-slice at pquant position `pq_x`.
        key : (None, pq_y)
            Gets the data `y`-slice at pquant position `pq_y`.
        key : (pq_x, pq_y)
            Gets the data at pquant position `(pq_x, pq_y)`.
        key : ((pq_x, pq_x_stop)) == ((pq_x, pq_x_stop), None)
            Gets a `x`-sliced matrix at pquant positions from
            `pq_x` to `pq_x_stop`.
        key : (None, (pq_y, pq_y_stop))
            Gets a `y`-sliced matrix at pquant positions from
            `pq_y` to `pq_y_stop`.
        key : ((pq_x, pq_x_stop), (pq_y, pq_y_stop))
            Gets an `x`- and `y`-sliced matrix with rectangular
            pquant positions `(pq_x, pq_y)` and
            `(pq_x_stop, pq_y_stop)`.

        Notes
        -----
        * Does not check for `key` validity.
        * The stop points `pq_x_stop` and `pq_y_stop` are included
          in the returned slice.
        * Slow performance because of coordinate transformation
          between indices and pquants. Implement calculations by
          directly accessing with indices.
        """
        # Initialize local variables
        pq_x, pq_y = key[0], None
        if len(key) == 2:
            pq_y = key[1]
        ind_x, ind_y = None, None
        # Find indices
        if pq_x is not None:
            if type(pq_x) == tuple:
                ind_x = (
                    self.var_rect.get_index_by_pquant(pq_x=pq_x[0])[0],
                    self.var_rect.get_index_by_pquant(pq_x=pq_x[1])[0]
                )
            else:
                ind_x = self.var_rect.get_index_by_pquant(pq_x=pq_x)[0]
        if pq_y is not None:
            if type(pq_y) == tuple:
                ind_y = (
                    self.var_rect.get_index_by_pquant(pq_y=pq_y[0])[1],
                    self.var_rect.get_index_by_pquant(pq_y=pq_y[1])[1]
                )
            else:
                ind_y = self.var_rect.get_index_by_pquant(pq_y=pq_y)[1]
        # Get val_data
        if ind_x is None:
            if type(ind_y) == tuple:
                return self.val_data[:, ind_y[0]:ind_y[1] + 1]
            else:
                return self.val_data[:, ind_y]
        elif ind_y is None:
            if type(ind_x) == tuple:
                return self.val_data[ind_x[0]:ind_x[1] + 1]
            else:
                return self.val_data[ind_x]
        else:   # ind are not None
            if type(ind_x) == tuple:
                if type(ind_y) == tuple:
                    return self.val_data[ind_x[0]:ind_x[1] + 1,
                                         ind_y[0]:ind_y[1] + 1]
                else:
                    return self.val_data[ind_x[0]:ind_x[1] + 1, ind_y]
            else:
                return self.val_data[ind_x, ind_y]

    def __str__(self):
        s = ""
        # var_x
        s += "{:s} ({:s}): ({:f}, {:f})\n".format(
            str(self.var_x_pquant), self.var_x_pquant.dtype,
            self.var_rect.pquant_bound_x[0], self.var_rect.pquant_bound_x[1],
        )
        # var_y
        s += "{:s} ({:s}): ({:f}, {:f})\n".format(
            str(self.var_y_pquant), self.var_y_pquant.dtype,
            self.var_rect.pquant_bound_y[0], self.var_rect.pquant_bound_y[1],
        )
        # val
        s += "{:s} ({:s}): ({:d}x{:d})\n".format(
            str(self.val_pquant), self.val_pquant.dtype,
            self.var_rect.ind_bound_x[1] - self.var_rect.ind_bound_x[0],
            self.var_rect.ind_bound_y[1] - self.var_rect.ind_bound_y[0]
        )
        # val_data
        s += str(self.val_data)
        return s

    #   #### Setter ####################

    def set_val_data(self,
                     val_data, val_pquant=None):
        """
        Sets the matrix and its metadata to be stored. Checks parameters.

        Parameters
        ----------
        val_data : `numpy.ndarray(2)`
            Matrix data.
        val_pquant : `data.types.pquant` or `None`, optional
            Physical quantity of values.
            `None` prevents overwriting previous val_pquant.

        Raises
        ------
        cfg.err.DTYPE_NDARRAY, cfg.err.INVAL_NONEMPTY, cfg.err.DTYPE_PQUANT
            If the parameters are invalid.
        """
        # Assign val_data
        ERR.assertion(ERR.DTYPE_NDARRAY,
                      type(val_data) == np.ndarray,
                      len(val_data.shape) == 2)
        ERR.assertion(ERR.INVAL_NONEMPTY, val_data.shape > (0, 0))
        self.val_data = val_data
        # Assign val descriptions
        if val_pquant is not None:
            ERR.assertion(ERR.DTYPE_PQUANT, type(val_pquant) == types.pquant)
            self.val_pquant = val_pquant
        self.var_rect.update_rect(self.val_data)

    def __setitem__(self, key, val):
        """Sets `val_data[key] = val` by index."""
        self.val_data[key] = val

    def set_var_data(self,
                     matrix_rect=None,
                     pq_pxsize_x=None, pq_pxsize_y=None,
                     pq_offset_x=None, pq_offset_y=None,
                     var_x_pquant=None, var_y_pquant=None):
        """
        Sets the variable rectangle and its metadata to be stored.
        Checks parameters.
        In all parameter cases, `None` prevents overwriting previously
        set (meta-) data. Rectangle settings are special: If any value
        (pxsize or offset on any axis) is to be changed, all four values
        must be explicitly set.

        Parameters
        ----------
        matrix_rect : `MatrixRect` or `None`, optional
            Matrix rectangle that is applied to the matrix data.
            If not `None`, overwrites any inputs to `pq_pxsize` and
            `pq_offset`.
        pq_pxsize_x, pq_pxsize_y : `float` or `None`, optional
            Pixel sizes in x and y directions in units of physical
            quantity.
        pq_offset_x, pq_offset_y : `float` or `None`, optional
            Matrix offsets in x and y directions in units of physical
            quantity.
        var_x_pquant, var_y_pquant : `types.pquant` or `None`, optional
            Physical quantity of variables on x, y axis.

        Raises
        ------
        cfg.err.DTYPE_FLOAT, cfg.err.INVAL_NONZERO, cfg.err.DTYPE_PQUANT
            If the parameters are invalid.
        """
        # Assign var_rect
        if type(matrix_rect) == MatrixRect:
            self.var_rect = matrix_rect
        else:
            if (pq_pxsize_x is not None and pq_pxsize_y is not None
                    and pq_offset_x is not None and pq_offset_y is not None):
                ERR.assertion(ERR.DTYPE_FLOAT, type(pq_pxsize_x) == float)
                ERR.assertion(ERR.DTYPE_FLOAT, type(pq_pxsize_y) == float)
                ERR.assertion(ERR.DTYPE_FLOAT, type(pq_offset_x) == float)
                ERR.assertion(ERR.DTYPE_FLOAT, type(pq_offset_y) == float)
                ERR.assertion(ERR.INVAL_NONZERO, pq_pxsize_x != 0)
                ERR.assertion(ERR.INVAL_NONZERO, pq_pxsize_y != 0)
                self.var_rect.set_rect_by_pixels(
                    self.val_data,
                    pq_pxsize_x=pq_pxsize_x, pq_pxsize_y=pq_pxsize_y,
                    pq_offset_x=pq_offset_x, pq_offset_y=pq_offset_y
                )
        # Assign val descriptions
        if var_x_pquant is not None:
            ERR.assertion(ERR.DTYPE_PQUANT, type(var_x_pquant) == types.pquant)
            self.var_x_pquant = var_x_pquant
        if var_y_pquant is not None:
            ERR.assertion(ERR.DTYPE_PQUANT, type(var_y_pquant) == types.pquant)
            self.var_y_pquant = var_y_pquant


###############################################################################


if __name__ == "__main__":

    # Create test data (image)
    data = np.arange(100, dtype="float64").reshape((10, 10))
    pq_data = types.pquant(
        name="intensity", unit="mW/cm2", dtype="float64"
    )
    pixel_size = 6.45
    pixel_offset = 0.0
    pq_x_pos = types.pquant(
        name="x position", unit="µm", dtype="float64"
    )
    pq_y_pos = types.pquant(
        name="y position", unit="µm", dtype="float64"
    )

    # Load into data structure
    mdata = MatrixData()
    mdata.set_val_data(data, val_pquant=pq_data)
    mdata.set_var_data(
        pq_pxsize_x=pixel_size, pq_pxsize_y=pixel_size,
        pq_offset_x=pixel_offset, pq_offset_y=pixel_offset,
        var_x_pquant=pq_x_pos, var_y_pquant=pq_y_pos
    )
    print(mdata)
