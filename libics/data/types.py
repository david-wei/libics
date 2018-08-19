# Package Imports
try:
    from . import addpath   # noqa
except(ImportError):
    import addpath          # noqa
import cfg.err as ERR
import util.serialization as ser


###############################################################################
# Physical Quantity
###############################################################################


class pquant(ser.DictSerialization, object):

    """
    Data type for physical quantities (name and unit).

    Parameters
    ----------
    name : `str`, optional
        Name of physical quantity.
    unit : `str` or `None`, optional
        Unit of physical quantity. None is interpreted as
        unitless quantity.
    dtype: `str`, optional
        Data type in which the physical values should be stored.
        Examples: `int64`, `float64` (default), `complex128`,
        `Sn` (string with byte length `n`)
    """

    def __init__(self,
                 name="", unit=None, dtype="float64"):
        self.name = name
        self.unit = unit
        self.dtype = dtype

    #   #### Setter ####################

    def set_name(self,
                 name):
        """
        Sets the name of the physical quantity.

        Parameters
        ----------
        name : `str`
            Name of physical quantity.

        Raises
        ------
        cfg.err.DTYPE_STR
            If the parameters are invalid.
        """
        ERR.assertion(ERR.DTYPE_STR, type(name) == str)
        self.name = name

    def set_unit(self,
                 unit):
        """
        Sets the unit of the physical quantity.

        Parameters
        ----------
        unit : `str` or `None`
            Unit of physical quantity. None is interpreted as
            unitless quantity.

        Raises
        ------
        cfg.err.DTYPE_STR
            If the parameters are invalid.
        """
        ERR.assertion(ERR.DTYPE_STR, type(unit) == str)
        self.unit = unit

    def set_dtype(self,
                  dtype):
        """
        Sets the data type in which values of the physical quantity
        should be stored.

        Parameters
        ----------
        dtype: `str`
            Data type in which the physical values should be stored.
            Examples: `int64`, `float64` (default), `complex128`,
            `Sn` (string with byte length `n`)

        Raises
        ------
        AssertionError
            If the parameters are invalid.
        """
        self.dtype = dtype

    def set_attr(self,
                 name, unit, dtype="float64"):
        """
        Sets name and unit of the physical quantity.

        Parameters
        ----------
        name : `str`
            Name of physical quantity.
        unit : `str`
            Unit of physical quantity.
        dtype: `str`, optional
            Data type in which the physical values should be stored.
            Examples: `int64`, `float64` (default), `complex128`,
            `Sn` (string with byte length `n`), `O` (Python object)

        Raises
        ------
        AssertionError
            If the parameters are invalid.
            Might modify stored data.
        """
        self.set_name(name)
        self.set_unit(unit)
        self.set_dtype(dtype)

    #   #### Operators #################

    def __str__(self):
        """Returns `"name [unit]"`."""
        s = self.name
        if self.unit is not None:
            s += " [" + self.unit + "]"
        return s

    def __eq__(self, other):
        """Returns an element-wise comparison."""
        return (
            self.name == other.name and
            self.unit == other.unit and
            self.dtype == other.dtype
        )

    #   #### Transformations ###########

    def invert(self):
        """
        Inverts the physical quantity.

        Examples
        --------
        >>> pq = pquant("position", "µm")
        >>> pq.invert()
        >>> pq
        "1/position [1/µm]"
        """
        if self.name[:2] == "1/":
            self.name = self.name[2:]
        else:
            self.name = "1/" + self.name
        if self.unit[:2] == "1/":
            self.unit = self.unit[2:]
        else:
            self.unit = "1/" + self.unit

    def divide(self, *divs):
        """
        Divides the current physical quantity by the passed pquants.

        Parameters
        ----------
        *divs : pquant
            Physical quantities by which current quantity is divided.

        Examples
        --------
        >>> pq = pquant("power", "mW")
        >>> d0 = pquant("x position", "µm")
        >>> d1 = pquant("y position", "mm")
        >>> pq.divide(d0, d1)
        >>> pq
        "power/x position/y position [mW/µm/mm]"
        """
        for div in divs:
            name = div.name
            unit = div.unit
            if name[:2] == "1/":
                name = name[1:]
            else:
                name = "/" + name
            if unit[:2] == "1/":
                unit = unit[1:]
            else:
                unit = "/" + unit
            self.name += name
            self.unit += unit
