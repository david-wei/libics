import collections
import numpy as np


###############################################################################


def NamedTuple(mapping):
    """
    Recursively converts mappings to nested named tuples.

    Named tuples are mapping objects (dicts) whose key-value pairs are
    accessible with attribute notation (e.g. object.attr_lvl0.attr_lvl1.key).
    Warning: converts all numeric keys to strings!

    Parameters
    ----------
    mapping : `collections.Mapping`
        Mapping object like dictionaries.

    Returns
    -------
    named_tuple : `collections.namedtuple`
        Nested named tuple.
    """

    if (
        isinstance(mapping, collections.Mapping)
        and not isinstance(mapping, ProtectedDict)
    ):
        for key, value in mapping.items():
            mapping[key] = NamedTuple(value)
        return cv_mapping_to_namedtuple(mapping)
    return mapping


def cv_mapping_to_namedtuple(mapping, name="NamedTuple"):
    """
    Converts a mapping to a named tuple (non-recursively).
    """
    return collections.namedtuple(name, mapping.keys())(**mapping)


class ProtectedDict(collections.UserDict):
    """
    Dictionary wrapper to make :py:class:`NamedTuple` not convert it to
    a named tuple (to keep it as `dict`).
    """


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
    """

    def __init__(
        self, allow_none=True, check_func=None, check_type=None,
        check_iter=None, check_min=None, check_max=None,
        assume_func=None
    ):
        self.allow_none = allow_none
        self.check_func = check_func
        self.check_type = check_type
        self.check_iter = check_iter
        self.check_min = check_min
        self.check_max = check_max
        self.assume_func = assume_func

    def __set_name__(self, owner, name):
        self.name = name

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
