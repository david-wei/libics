import collections
from importlib import import_module


###############################################################################


def get_class_from_string(path):
    """
    Gets the class specified by the given string.

    Parameters
    ----------
    path : `str`
        Python-attribute-style string specifying class, e.g.
        `libics.data.arraydata.ArrayData`.

    Returns
    -------
    _cls : `class`
        Requested class object.
    """
    module_path, _, class_name = path.rpartition('.')
    try:
        mod = import_module(module_path)
        _cls = getattr(mod, class_name)
        return _cls
    except (AttributeError, ValueError):
        raise ImportError("invalid class ({:s})".format(str(path)))


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
