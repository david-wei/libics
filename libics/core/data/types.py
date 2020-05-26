import numpy as np


###############################################################################
# Primitive containers
###############################################################################


NO_NAME = "N/A"


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

    def __init__(self, name=NO_NAME, symbol=None, unit=None):
        self.name = name
        self.symbol = symbol
        self.unit = unit

    # ++++++++++++++++
    # Unary operations
    # ++++++++++++++++

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

    # +++++++++++++++++
    # Binary operations
    # +++++++++++++++++

    def _is_quantity(self, other):
        return isinstance(other, type(self))

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

    Supports common arithmetic operations (`+, -, *, /, //, %, **`)
    and comparison operations (`==, !=, <, <=, >, >=`)
    with other `ValQuantity` objects or numeric values.
    Supports numeric casting into `int, float, complex`.

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

    def __init__(self, name=NO_NAME, symbol=None, unit=None, val=None):
        super().__init__(name=name, symbol=symbol, unit=unit)
        self.val = val

    # ++++++++++++++++
    # Unary operations
    # ++++++++++++++++

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

    def __int__(self):
        return int(self.val)

    def __float__(self):
        return float(self.val)

    def __complex__(self):
        return complex(self.val)

    def __neg__(self):
        _name, _symbol, _unit = self.name, self.symbol, self.unit
        _val = -self.val
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    # +++++++++++++++++
    # Binary operations
    # +++++++++++++++++

    ALL_OPS = ["+", "-", "*", "/", "//", "%", "**", "&"]
    LINEAR_OPS = ["+", "-", "%", "&"]
    COMBINE_OPS = ["*"]
    CANCEL_OPS = ["/", "//"]
    NUMERIC_OPS = ["**"]
    STR_OP = {i: i for i in ALL_OPS}
    MATHSTR_OP = {
        "+": "+", "-": "-", "*": "\\cdot", "/": "/", "//": "\\mathrm{{div}}",
        "%": "\\mathrm{{mod}}", "**": "^", "&": "&"
    }

    @staticmethod
    def _brk(s):
        """
        Surrounds with brackets if string `s` contains a space.
        """
        if " " in s:
            return "({:s})".format(s)
        else:
            return s

    def _get_common_name(self, other, op="&", rev=False):
        _name = NO_NAME
        if self.name == NO_NAME:
            _name = other.name
        elif other.name == NO_NAME:
            _name = self.name
        else:
            _a = [self._brk(self.name), self.STR_OP[op], self._brk(other.name)]
            _name = "{:s} {:s} {:s}".format(*(reversed(_a) if rev else _a))
        return _name

    def _get_common_symbol(self, other, self_first=True, op="&", rev=False):
        _symbol = None
        if self.symbol is None:
            _symbol = other.symbol
        elif other.symbol is None:
            _symbol = self.symbol
        else:
            _a = [self._brk(self.symbol), self.MATHSTR_OP[op],
                  self._brk(other.symbol)]
            _symbol = "{:s} {:s} {:s}".format(*(reversed(_a) if rev else _a))
        return _symbol

    def _get_common_unit(self, other, op="&", rev=False):
        _unit = None
        if op in self.LINEAR_OPS:
            if self.unit is None:
                _unit = other.unit
            elif other.unit is None:
                _unit = self.unit
            elif self.unit != other.unit:
                _a = [self._brk(self.unit), self.STR_OP[op],
                      self._brk(other.unit)]
                raise ValueError("invalid units: {:s} {:s} {:s}"
                                 .format(*(reversed(_a) if rev else _a)))
            else:
                _unit = self.unit
        elif op in self.COMBINE_OPS:
            if self.unit is None:
                _unit = other.unit
            elif other.unit is None:
                _unit = self.unit
            else:
                _a = [self._brk(self.unit), self.STR_OP[op],
                      self._brk(other.unit)]
                _unit = "{:s} {:s} {:s}".format(*(reversed(_a) if rev else _a))
        elif op in self.CANCEL_OPS:
            if self.unit is None:
                if rev:
                    _unit = other.unit
                else:
                    _a = [self.STR_OP[op], self._brk(other.unit)]
                    _unit = "1 {:s} {:s}".format(_a)
            elif other.unit is None:
                if rev:
                    _unit = self.unit
                else:
                    _a = [self.STR_OP[op], self._brk(self.unit)]
                    _unit = "1 {:s} {:s}".format(_a)
            elif self.unit == other.unit:
                _unit = None
            else:
                _a = [self._brk(self.unit), self.STR_OP[op],
                      self._brk(other.unit)]
                _unit = "{:s} {:s} {:s}".format(*(reversed(_a) if rev else _a))
        elif op in self.NUMERIC_OPS:
            _unit = None
        return _unit

    def __eq__(self, other):
        if self._is_quantity(other):
            return self.val == other.val and super().__eq__(other)
        else:
            return self.val == other

    def __ne__(self, other):
        if self._is_quantity(other):
            return self.val != other.val or super().__ne__(other)
        else:
            return self.val != other

    def __lt__(self, other):
        if self._is_quantity(other):
            return self.val < other.val
        else:
            return self.val < other

    def __le__(self, other):
        if self._is_quantity(other):
            return self.val <= other.val
        else:
            return self.val <= other

    def __gt__(self, other):
        if self._is_quantity(other):
            return self.val > other.val
        else:
            return self.val > other

    def __ge__(self, other):
        if self._is_quantity(other):
            return self.val >= other.val
        else:
            return self.val >= other

    def __add__(self, other):
        if self._is_quantity(other):
            _name = self._get_common_name(other, op="+")
            _symbol = self._get_common_symbol(other, op="+")
            _unit = self._get_common_unit(other, op="+")
            _val = self.val + other.val
        else:
            _name = self.name
            _symbol = self.symbol
            _unit = self.unit
            _val = self.val + other
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __sub__(self, other):
        if self._is_quantity(other):
            _name = self._get_common_name(other, op="-")
            _symbol = self._get_common_symbol(other, op="-")
            _unit = self._get_common_unit(other, op="-")
            _val = self.val - other.val
        else:
            _name = self.name
            _symbol = self.symbol
            _unit = self.unit
            _val = self.val - other
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __mul__(self, other):
        if self._is_quantity(other):
            _name = self._get_common_name(other, op="*")
            _symbol = self._get_common_symbol(other, op="*")
            _unit = self._get_common_unit(other, op="*")
            _val = self.val * other.val
        else:
            _name = self.name
            _symbol = self.symbol
            _unit = self.unit
            _val = self.val * other
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __truediv__(self, other):
        if self._is_quantity(other):
            _name = self._get_common_name(other, op="/")
            _symbol = self._get_common_symbol(other, op="/")
            _unit = self._get_common_unit(other, op="/")
            _val = self.val / other.val
        else:
            _name = self.name
            _symbol = self.symbol
            _unit = self.unit
            _val = self.val / other
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __floordiv__(self, other):
        if self._is_quantity(other):
            _name = self._get_common_name(other, op="//")
            _symbol = self._get_common_symbol(other, op="//")
            _unit = self._get_common_unit(other, op="//")
            _val = self.val // other.val
        else:
            _name = self.name
            _symbol = self.symbol
            _unit = self.unit
            _val = self.val // other
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __mod__(self, other):
        if self._is_quantity(other):
            _name = self._get_common_name(other, op="%")
            _symbol = self._get_common_symbol(other, op="%")
            _unit = self._get_common_unit(other, op="%")
            _val = self.val % other.val
        else:
            _name = self.name
            _symbol = self.symbol
            _unit = self.unit
            _val = self.val % other
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __pow__(self, other):
        if self._is_quantity(other):
            _name = self._get_common_name(other, op="**")
            _symbol = self._get_common_symbol(other, op="**")
            _unit = self._get_common_unit(other, op="**")
            _val = self.val**other.val
        else:
            _name = self.name
            _symbol = self.symbol
            _unit = None
            _val = self.val**other
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        _name = self.name
        _symbol = self.symbol
        _unit = self.unit
        _val = other - self.val
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        _name = self.name
        _symbol = self.symbol
        _unit = self.unit
        _val = other / self.val
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __rfloordiv__(self, other):
        _name = self.name
        _symbol = self.symbol
        _unit = self.unit
        _val = other // self.val
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __rmod__(self, other):
        _name = self.name
        _symbol = self.symbol
        _unit = self.unit
        _val = other % self.val
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __rpow__(self, other):
        _name = self.name
        _symbol = self.symbol
        _unit = self.unit
        _val = other**self.val
        return ValQuantity(name=_name, symbol=_symbol, unit=_unit, val=_val)

    def __iadd__(self, other):
        if self._is_quantity(other):
            self.name = self._get_common_name(other, op="+")
            self.symbol = self._get_common_symbol(other, op="+")
            self.unit = self._get_common_unit(other, op="+")
            self.val += other.val
        else:
            self.val += other
        return self

    def __isub__(self, other):
        if self._is_quantity(other):
            self.name = self._get_common_name(other, op="-")
            self.symbol = self._get_common_symbol(other, op="-")
            self.unit = self._get_common_unit(other, op="-")
            self.val -= other.val
        else:
            self.val -= other
        return self

    def __imul__(self, other):
        if self._is_quantity(other):
            self.name = self._get_common_name(other, op="*")
            self.symbol = self._get_common_symbol(other, op="*")
            self.unit = self._get_common_unit(other, op="*")
            self.val *= other.val
        else:
            self.val *= other
        return self

    def __itruediv__(self, other):
        if self._is_quantity(other):
            self.name = self._get_common_name(other, op="/")
            self.symbol = self._get_common_symbol(other, op="/")
            self.unit = self._get_common_unit(other, op="/")
            self.val /= other.val
        else:
            self.val /= other
        return self

    def __ifloordiv__(self, other):
        if self._is_quantity(other):
            self.name = self._get_common_name(other, op="//")
            self.symbol = self._get_common_symbol(other, op="//")
            self.unit = self._get_common_unit(other, op="//")
            self.val //= other.val
        else:
            self.val //= other
        return self

    def __imod__(self, other):
        if self._is_quantity(other):
            self.name = self._get_common_name(other, op="%")
            self.symbol = self._get_common_symbol(other, op="%")
            self.unit = self._get_common_unit(other, op="%")
            self.val %= other.val
        else:
            self.val %= other
        return self

    def __ipow__(self, other):
        if self._is_quantity(other):
            self.name = self._get_common_name(other, op="**")
            self.symbol = self._get_common_symbol(other, op="**")
            self.unit = self._get_common_unit(other, op="**")
            self.val **= other.val
        else:
            self.unit = None
            self.val **= other
        return self


###############################################################################
# Functional descriptors
###############################################################################


class ValCheckDesc:

    """
    Descriptor class for validity-checked attributes.

    Also provides an interface for data structure assumptions.
    Raises `ValueError` if invalid.

    Parameters
    ----------
    allow_none : `bool`
        Flag whether to allow the `None` value regardless of other validity
        checks.
    check_func : `callable` or `None`
        Call signature: `val_check(new_val)->bool`.
        Should return `True` if valid and `False` if not.
    check_type : `class` or `None`
        Valid classes an object can be an instance of.
    check_iter : `iterable` or `None`
        Iterable containing valid values.
    check_min, check_max : `object` or `None`
        Objects implementing the `<=, >=` operators, which are checked against.
        Allows for vectorized values.
    assume_func : `callable` or `None`
        Function that changes the input into a data format the value should
        be stored in. Call signature: `assume_func(new_val)->object`.
        Is called after the check functions.
    add_io_key : `bool`
        Flag whether to include descriptor when serializing the owning class
        using the functions in `libics.file.io`.
    """

    def __init__(
        self, allow_none=True, check_func=None, check_type=None,
        check_iter=None, check_min=None, check_max=None,
        assume_func=None, add_io_key=True
    ):
        self.allow_none = allow_none
        self.check_func = check_func
        self.check_type = check_type
        self.check_iter = check_iter
        self.check_min = check_min
        self.check_max = check_max
        self.assume_func = assume_func
        self.add_io_key = add_io_key

    def __set_name__(self, owner, name):
        self.name = name
        import libics.core.io
        if self.add_io_key and isinstance(owner, libics.core.io.FileBase):
            owner.SER_KEYS.add(name)

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if value is None and self.allow_none:
            pass
        else:
            if self.check_func is not None and not self.check_func(value):
                self._handle_invalid(value)
            if (
                self.check_type is not None
                and not isinstance(value, self.check_type)
            ):
                self._handle_invalid(value)
            if self.check_iter is not None and value not in self.check_iter:
                self._handle_invalid(value)
            if (
                self.check_min is not None
                and not np.all(self.check_min <= value)
            ):
                self._handle_invalid(value)
            if (
                self.check_max is not None
                and not np.all(self.check_max >= value)
            ):
                self._handle_invalid(value)
        if self.assume_func is not None:
            value = self.assume_func(value)
        instance.__dict__[self.name] = value

    def _handle_invalid(self, value):
        raise ValueError(
            "invalid {:s} value ({:s})".format(self.name, str(value))
        )
