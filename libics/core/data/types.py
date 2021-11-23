import collections
import functools
import numpy as np
import operator
from xxhash import xxh64_intdigest


###############################################################################
# Operators
###############################################################################


BINARY_OPS_NUMPY = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.true_divide,
    "//": np.floor_divide,
    "%": np.mod,
    "**": np.power,
    "&": np.bitwise_and,
    "|": np.bitwise_or,
    "^": np.bitwise_xor,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
    ">=": np.greater_equal,
    ">": np.greater,
    "<<": np.left_shift,
    ">>": np.right_shift
}
UNARY_OPS_NUMPY = {
    "~": np.invert,
    "-": np.negative
}


###############################################################################
# Hashing
###############################################################################


def hash_combine_ordered(*hval):
    """
    Combines multiple hash values (asymmetric).
    """
    if len(hval) == 1:
        return hval[0]
    else:
        return hash(hval)


def hash_combine_xor(*hval):
    """
    Combines multiple hash values by xor (symmetric).
    """
    return functools.reduce(operator.xor, hval)


def hash_libics(val):
    """
    Implements a hash function that modifies built-in `hash` as follows:

    * Stable hash of `bytes`-like objects via `xxh64`.
    * Support for `numpy` arrays via `xxh64`.
    * Recursively hashes iterables including `dict` and `list`.
    """
    if isinstance(val, np.ndarray):
        if val.dtype == complex:
            return hash_combine_ordered(
                xxh64_intdigest(np.real(val)),
                xxh64_intdigest(np.imag(val))
            )
        else:
            return xxh64_intdigest(val)
    elif isinstance(val, AttrHashBase):
        return val._hash_libics()
    elif isinstance(val, collections.Mapping):
        return hash_combine_xor(*(
            hash_combine_ordered(hash_libics(k), hash_libics(val[k]))
            for k in val
        ))
    else:
        try:
            return xxh64_intdigest(val)
        except TypeError:
            pass
        try:
            return hash(val)
        except TypeError:
            return hash_combine_ordered(*(hash_libics(_v) for _v in val))


class AttrHashBase:

    """
    Base class for attribute hashing.

    For a given set of attributes (:py:attr:`HASH_KEYS`), constructs a hash
    value that has the following properties:

    * Is unordered in the hash keys.
    * Includes mapping between attribute name and value.
    * Includes fully qualified class name.
    * The attribute values must have `__hash__()` implemented.

    Examples
    --------
    Subclassing :py:class:`AttrHashBase`:

    >>> class MyClass(AttrHashBase):
    ...     HASH_KEYS = AttrHashBase.HASH_KEYS | {"attr0", "attr1"}
    ...     def __init__(self):
    ...         self.attr0 = "asdf"
    ...         self.attr1 = 1234
    >>> my_object = MyClass()
    >>> hex(hash(my_object))
    '0x1b6ce7488edaa385'

    Note that if subclassing :py:class:`AttrHashBase` and overriding the
    `__eq__` method, the `__hash__` method has to be explicitly reimplemented:

    >>> class Sub(AttrHashBase):
    ...     def __eq__(self, other):
    ...         return id(self) == id(other)
    ...     def __hash__(self):
    ...         return super().__hash__()
    """

    HASH_KEYS = set()

    def _hash_libics(self):
        hash_name = hash_libics(
            self.__class__.__module__ + "." + self.__class__.__name__
        )
        hash_vals = (
            hash_combine_ordered(
                hash_libics(k), hash_libics(getattr(self, k))
            ) for k in self.HASH_KEYS
        )
        return hash_combine_xor(hash_name, *hash_vals)

    def __hash__(self):
        return self._hash_libics()


###############################################################################
# Primitive containers
###############################################################################


NO_NAME = "N/A"


class Quantity(AttrHashBase):

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
        super().__init__()
        self.name = name
        self.symbol = symbol
        self.unit = unit

    __LIBICS_IO__ = True
    SER_KEYS = {"name", "symbol", "unit"}
    HASH_KEYS = AttrHashBase.HASH_KEYS | SER_KEYS

    def attributes(self):
        """Implements :py:meth:`libics.core.io.FileBase.attributes`."""
        return {k: getattr(self, k) for k in self.SER_KEYS}

    # ++++++++++++++++
    # Unary operations
    # ++++++++++++++++

    def __str__(self):
        s = self.name
        if self.symbol is not None:
            s += " " + self.symbol
        if self.unit is not None:
            s += " (" + self.unit + ")"
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
            s += r" ($\mathregular{" + self.unit + r"}$)"
        return s

    def labelstr(self, math=True, **kwargs):
        if self.has_name():
            if math is True:
                return self.mathstr(**kwargs)
            else:
                return str(self)
        else:
            return None

    def has_name(self):
        return self.name != NO_NAME

    # +++++++++++++++++
    # Binary operations
    # +++++++++++++++++

    def _is_quantity(self, other):
        return isinstance(other, type(self))

    def __hash__(self):
        return super().__hash__()

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

    SER_KEYS = Quantity.SER_KEYS | {"val"}
    HASH_KEYS = Quantity.HASH_KEYS | SER_KEYS

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
                s += " (" + self.unit + ")"
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
                s += " (" + str_unit + ")"
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
                    _unit = "1 {:s} {:s}".format(*_a)
            elif other.unit is None:
                if rev:
                    _unit = self.unit
                else:
                    _a = [self.STR_OP[op], self._brk(self.unit)]
                    _unit = "1 {:s} {:s}".format(*_a)
            elif self.unit == other.unit:
                _unit = None
            else:
                _a = [self._brk(self.unit), self.STR_OP[op],
                      self._brk(other.unit)]
                _unit = "{:s} {:s} {:s}".format(*(reversed(_a) if rev else _a))
        elif op in self.NUMERIC_OPS:
            _unit = None
        return _unit

    def __hash__(self):
        return super().__hash__()

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


###############################################################################
# Structures
###############################################################################


class AttrDict(dict):

    """
    Dictionary container supporting recursive attribute access of items.

    Parameters
    ----------
    rec : `bool`
        Whether given sub-dictionaries should be recursively converted
        to `AttrDict` objects.

    Notes
    -----
    * If a requested attribute does not exist, creates an empty `AttrDict`
      object to prevent empty attributes. This is only implemented for public
      attribute names, i.e. fails when the name starts with `"_"`.
    * If multiple sequential dots (e.g. `"abc..def"`) are in a key, it
      is interpreted as single string (instead of attributes).
      This behavior was chosen to support ellipses (`"..."`) in strings.

    Examples
    --------
    >>> my_dict = {"a": "A", "b": {"c": "C"}}
    >>> my_attrdict = AttrDict(my_dict)
    >>> my_attrdict
    {'a': 'A', 'b': {'c': 'C'}}
    >>> my_attrdict["b"]["c"] == my_attrdict.b.c
    True
    >>> my_attrdict.d.e.f = "nested_val"
    >>> my_attrdict
    {'a': 'A', 'b': {'c': 'C'}, 'd': {'e': {'f': 'nested_val'}}}
    """

    def __init__(self, *args, rec=True, **kwargs):
        # Update constructor dict by kwarg items
        if len(kwargs) > 0:
            if len(args) == 0:
                args = ({},)
            args[0].update(kwargs)
        # Recursive constructor
        if rec is True:
            super().__init__()
            if len(args) > 0:
                d = args[0]
                for k, v in d.items():
                    if isinstance(v, dict):
                        self[k] = AttrDict(v)
                    else:
                        self[k] = v
        # Non-recursive constructor
        else:
            super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError(f"invalid key type {repr(key)} (must be str)")
        if ("." not in key) or (".." in key):
            return super().__getitem__(key)
        else:
            key, subkey = key.split(".", 1)
            return super().__getitem__(key)[subkey]

    def __setitem__(self, key, val):
        if not isinstance(key, str):
            raise KeyError(f"invalid key type {repr(key)} (must be str)")
        if ("." not in key) or (".." in key):
            super().__setitem__(key, val)
        else:
            key, subkey = key.split(".", 1)
            if key not in self:
                super().__setitem__(key, AttrDict())
            super().__getitem__(key)[subkey] = val

    def __getattr__(self, key):
        if key not in self and key[0] != "_":
            self[key] = AttrDict()
        elif key not in self:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )
        return self[key]

    def __setattr__(self, key, val):
        if key not in self and key[0] == "_":
            self.__dict__[key] = val
        else:
            self[key] = val
