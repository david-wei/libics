###############################################################################
# Primitive containers
###############################################################################


class Quantity(object):

    """
    Data type for physical quantities (name and unit).

    Parameters
    ----------
    name : `str`, optional
        Name of physical quantity.
    symbol : str or None, optional
        Symbol of variable.
    unit : `str` or `None`, optional
        Unit of physical quantity. None is interpreted as
        unitless quantity.
    """

    def __init__(self,
                 name="N/A", symbol=None, unit=None,
                 cls_name="Quantity"):
        super().__init__(pkg_name="libics", cls_name=cls_name)
        self.name = name
        self.symbol = symbol
        self.unit = unit

    def __str__(self):
        s = self.name
        if self.symbol is not None:
            s += " " + self.symbol
        if self.unit is not None:
            s += " [" + self.unit + "]"
        return s

    def __repr__(self):
        return str(type(self)) + ": " + str(self)

    def mathstr(self, name=True, symbol=True, unit=True, **kwargs):
        s = ""
        if name:
            s += self.name
        if symbol and self.symbol is not None:
            s += " $" + self.symbol + "$"
        if unit and self.unit is not None:
            s += r" [$\mathregular{" + self.unit + r"}$]"
        return s

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.symbol == other.symbol
            and self.unit == other.unit
        )

    def __ne__(self, other):
        return (
            self.name != other.name
            or self.symbol != other.symbol
            or self.unit != other.unit
        )


class ValQuantity(Quantity):

    """
    Data type for physical quantities with values.

    Parameters
    ----------
    name : `str`, optional
        Name of physical quantity.
    symbol : str or None, optional
        Symbol of variable.
    unit : `str` or `None`, optional
        Unit of physical quantity. None is interpreted as
        unitless quantity.
    val
        Value of physical quantity.
    """

    def __init__(self, name="N/A", symbol=None, unit=None, val=None):
        super().__init__(
            name=name, symbol=symbol, unit=unit, cls_name="ValQuantity"
        )
        self.val = val

    def __str__(self):
        s = self.name
        if self.symbol is not None:
            s += " " + self.symbol
            if self.val is not None:
                s += " ="
        elif self.val is not None:
            s += ":"
        if self.val is not None:
            s += " " + str(self.val)
        if self.unit is not None:
            if self.val is not None:
                s += " " + self.unit
            else:
                s += " [" + self.unit + "]"
        return s

    def mathstr(self, name=True, symbol=True, unit=True,
                val=True, val_format=None, unit_math=False):
        s = ""
        if name:
            s += self.name
        if symbol and self.symbol is not None:
            s += " $" + self.symbol + "$"
            if val and self.val is not None:
                s += " ="
        if val and self.val is not None:
            if val_format is None:
                s += " $" + str(self.val) + "$"
            else:
                s += " $" + val_format.format(self.val) + "$"
        if unit and self.unit is not None:
            str_unit = ("$" + self.unit + "$") if unit_math else self.unit
            if val and self.val is not None:
                s += " " + str_unit
            else:
                s += " [" + str_unit + "]"
        return s

    def __eq__(self, other):
        return (
            self.val == other.val
            and super().__eq__(other)
        )

    def __ne__(self, other):
        return (
            self.val != other.val
            or super().__ne__(other)
        )

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __gt__(self, other):
        return self.val > other.val

    def __ge__(self, other):
        return self.val >= other.val


class FlaggedType:

    """
    Container for values with boolean flag.

    Provides a validity checker (range or subset based).
    Implements operators mathematical operators. On combinatory operation
    (e.g. `+`), flags are combined as `or` and conditions are dropped (`None`).

    Parameters
    ----------
    val
        Value to be stored.
    flag : bool
        Boolean flag.
    cond : tuple or list or None
        Value validity condition.
        tuple with length 2: range from min[0] to max[1].
        list: discrete set of allowed values.
        None: no validity check.
    """

    def __init__(self, val, flag=False, cond=None):
        self._val = None
        self._cond = None
        self.val = val
        self.flag = flag

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, v):
        if self._cond is not None:
            if type(self._cond) == tuple:
                if v < self._cond[0] or v > self._cond[1]:
                    raise ValueError("libics.util.types.FlaggedType.val: {:s}"
                                     .format(str(v)))
            else:
                if v not in self._cond:
                    raise ValueError("libics.util.types.FlaggedType.val: {:s}"
                                     .format(str(v)))
        self._val = v

    @property
    def cond(self):
        return self._cond

    @cond.setter
    def cond(self, c):
        if (c is None or type(c) == list or
                (type(c) == tuple and len(c) == 2 and c[0] <= c[1])):
            self._cond = c
        else:
            raise TypeError("libics.util.types.FlaggedType.cond: {:s}"
                            .format(str(c)))

    def invert(self):
        self.flag = not self.flag
        return self.flag

    def assign(self, other, diff_flag=False):
        """
        Copies the attributes of another `FlaggedType` object into the current
        object.

        Parameters
        ----------
        other : FlaggedType
            The object to be copied.
        diff_flag : bool
            Whether to copy the flag state or the differential flag
            state.
            `False`:
                Copies the flag state.
            `True`:
                Compares the values and sets the flag to
                `self.val != other.val`.
        """
        if diff_flag:
            self.flag = (self.val != other.val)
        else:
            self.flag = other.flag
        self.cond = other.cond
        self.val = other.val

    def set_val(self, val, diff_flag=True):
        """
        Sets the value without changing the condition.

        Parameters
        ----------
        val
            New value of flagged type.
        diff_flag : bool
            Whether to set a differential flag.
            `False`:
                Keeps the current flag state.
            `True`:
                Compares the values and sets the flag to
                `self.val != val`.
        """
        _old_val = self.val
        self.val = val
        if diff_flag:
            self.flag = (val != _old_val)

    def copy(self):
        """
        Returns an independent copy of itself.
        """
        return FlaggedType(self.val, flag=self.flag, cond=self.cond)

    def __eq__(self, other):
        return self.val == other.val

    def __ne__(self, other):
        return self.val != other.val

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __gt__(self, other):
        return self.val > other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __add__(self, other):
        return FlaggedType(
            self.val + other.val,
            flag=(self.flag or other.flag)
        )

    def __sub__(self, other):
        return FlaggedType(
            self.val - other.val,
            flag=(self.flag or other.flag)
        )

    def __mul__(self, other):
        return FlaggedType(
            self.val * other.val,
            flag=(self.flag or other.flag)
        )

    def __truediv__(self, other):
        return FlaggedType(
            self.val / other.val,
            flag=(self.flag or other.flag)
        )

    def __pow__(self, other):
        return FlaggedType(
            self.val**other.val,
            flag=(self.flag or other.flag)
        )

    def __neg__(self):
        return FlaggedType(-self.val, flag=self.flag, cond=self.cond)

    def __int__(self):
        return int(self.val)

    def __float__(self):
        return float(self.val)

    def __str__(self):
        return str(self.val)
