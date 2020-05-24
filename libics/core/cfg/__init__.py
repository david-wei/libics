import collections
import numpy as np
import configparser
import json

from libics.file import io


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


INI_DEFAULT_SECTION = configparser.DEFAULTSECT


class CfgBase(io.FileBase):

    """
    Configuration file base class.

    Attributes must be of built-in/primitive types or (subclasses of)
    `CfgBase`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_cfg_depth(self):
        """
        Gets the configuration nest level.
        """
        max_depth = 0
        for k in self.SER_KEYS:
            v = getattr(self, k)
            if isinstance(v, CfgBase):
                _depth = v._get_cfg_depth()
                if max_depth < _depth:
                    max_depth = _depth
        return max_depth

    def _to_dict(self, serialize_all=False):
        """
        Serializes the serializable attributes to a dict.

        Parameters
        ----------
        serialize_all : `bool`
            Flag whether to serialize all attributes or only the
            ones stated in `SER_KEYS`.

        Returns
        -------
        d : `dict`
            Serialized attributes.
        """
        if serialize_all:
            d = {k: v for k, v in self.__dict__.items()}
        else:
            d = self.attributes()
        for k, v in d.items():
            if isinstance(v, CfgBase):
                d[k] = v._to_dict()
        return d

    def _from_dict(self, d):
        """
        Sets attributes according to the given dict.

        Parameters
        ----------
        d : `dict`
            Serialized attributes.
        """
        for k, v in d.items():
            if isinstance(v, dict):
                if k not in self.__dict__:
                    setattr(self, k, CfgBase())
                getattr(self, k)._from_dict(v)
            else:
                setattr(self, k, v)

    def save_cfg(self, file_path, fmt=None, serialize_all=False, **kwargs):
        """
        Saves the configuration object to a text file.

        In contrast to the :py:meth:`save` method, this method does not
        save any metadata.

        Parameters
        ----------
        file_path : `str`
            Save file path.
        fmt : `str`
            File format: `"json"`, `"ini"`, `"yml"`.
        serialize_all : `bool`
            Flag whether to serialize all attributes or only the
            ones stated in `SER_KEYS`.
        **kwargs
            Keyword arguments passed to parsers. Notable arguments include:
            `"json"`: `indent`.
        """
        fmt = io.get_file_format(file_path, fmt=fmt)
        d = self._to_dict(serialize_all=serialize_all)
        if "json" in fmt:
            with open(file_path, "w") as f:
                json.dump(d, f, **kwargs)
        elif "ini" in fmt:
            _cfg_depth = self._get_cfg_depth()
            if _cfg_depth > 1:
                raise ValueError("maximum ini file depth exceeded ({:d})"
                                 .format(_cfg_depth))
            # Depth == 1 required for ini file, thus create dummy section
            dd = d.copy()
            d[INI_DEFAULT_SECTION] = {}
            for k, v in dd.items():
                if not isinstance(v, dict):
                    d[INI_DEFAULT_SECTION][k] = v
                    del d[k]
            if len(d[INI_DEFAULT_SECTION]) == 0:
                del d[INI_DEFAULT_SECTION]
            # Write ini file
            cp = configparser.ConfigParser(**kwargs)
            cp.read_dict(self._to_dict())
            with open(file_path, "w") as f:
                cp.write(f)
        elif "yml" in fmt:
            raise NotImplementedError("yml format not yet implemented")
        else:
            raise ValueError("invalid format ({:s})".format(fmt))

    def load_cfg(self, file_path, fmt=None, **kwargs):
        """
        Loads a configuration text file to object.

        In contrast to the :py:meth:`load` method, this method assumes no
        stored metadata.

        Parameters
        ----------
        file_path : `str`
            Save file path.
        fmt : `str`
            File format: `"json"`, `"ini"`, `"yml"`.
        **kwargs
            Keyword arguments passed to the parsers. Notable arguments include:
            `"ini"`: `interpolation`, `strict`.
        """
        fmt = io.get_file_format(file_path, fmt=fmt)
        if "json" in fmt:
            with open(file_path, "r") as f:
                d = json.load(f, **kwargs)
        elif "ini" in fmt:
            cp = configparser.ConfigParser(**kwargs)
            with open(file_path, "r") as f:
                cp.read_file(f)
            d = {s: dict(cp.items(s)) for s in cp.sections()}
            # Parse dummy section
            if INI_DEFAULT_SECTION in d:
                dd = d.copy()
                for k, v in dd[INI_DEFAULT_SECTION].items():
                    d[k] = v
                del d[INI_DEFAULT_SECTION]
        elif "yml" in fmt:
            raise NotImplementedError("yml format not yet implemented")
        else:
            raise ValueError("invalid format ({:s})".format(fmt))
        self._from_dict(d)


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
        if self.add_io_key and isinstance(owner, io.FileBase):
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
