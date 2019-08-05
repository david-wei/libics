# Package Imports
from libics.file import hdf
from libics.util import InheritMap


###############################################################################


@InheritMap(map_key=("libics", "Quantity"))
class Quantity(hdf.HDFBase):

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


@InheritMap(map_key=("libics", "ValQuantity"))
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
