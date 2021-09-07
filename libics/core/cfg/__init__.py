import collections
import configparser
import json

from libics.env import logging
from libics.core import io


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

    LOGGER = logging.get_logger("libics.core.cfg.CfgBase")

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_cfg_depth(self, serialize_all=False):
        """
        Gets the configuration nest level.
        """
        max_depth = 0
        for k in self.__dict__.keys() if serialize_all else self.SER_KEYS:
            v = getattr(self, k)
            if isinstance(v, CfgBase):
                _depth = 1 + v._get_cfg_depth(serialize_all=serialize_all)
                if max_depth < _depth:
                    max_depth = _depth
        return max_depth

    def __getitem__(self, key):
        keys = str.split(key, ".")
        obj = getattr(self, keys[0])
        if len(keys) > 1:
            obj = obj[".".join(keys[1:])]
        return obj

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
                d[k] = v._to_dict(serialize_all=serialize_all)
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

    def __str__(self):
        d = self._to_dict(serialize_all=False)
        if len(d) == 0:
            d = self._to_dict(serialize_all=True)
        s = str(d)
        return s

    def __repr__(self):
        return f"<'{self.__class__.__name__}' at {hex(id(self))}>\n{str(self)}"

    def map_recursive(self, item_func=None, cfg_func=None, use_all=False):
        """
        Applies a function recursively on all configuration items.

        Parameters
        ----------
        item_func : `callable`
            Function which is recursively called on configuration items
            Sets its value to the function return value.
            Call signature: `func(key, old_val)->new_val`.
        cfg_func : `callable`
            Function which is called on attributes which themselves are
            `CfgBase` objects.
            If the return value is `False`, recursively
            calls the `map_recursive` method on the attribute.
            If the return value is `True`, continues without further
            handling of the attribute.
            Call signature: `func(key, sub_cfg)->bool`.
        use_all : `bool`
            Flag whether to map all attributes or only the
            ones stated in `SER_KEYS`.
        """
        if use_all:
            d = {k: v for k, v in self.__dict__.items()}
        else:
            d = self.attributes()
        for k, v in d.items():
            if isinstance(v, CfgBase):
                call_item_func = True
                if cfg_func is not None:
                    ret = cfg_func(k, getattr(self, k))
                    if ret is True:
                        call_item_func = False
                if call_item_func:
                    getattr(self, k).map_recursive(item_func, use_all=use_all)
            elif item_func is not None:
                setattr(self, k, item_func(k, v))

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
            _cfg_depth = self._get_cfg_depth(serialize_all=serialize_all)
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
            cp.read_dict(d)
            with open(file_path, "w") as f:
                cp.write(f)
        elif "yml" in fmt:
            raise NotImplementedError("yml format not yet implemented")
        else:
            raise ValueError("invalid format ({:s})".format(fmt))

    def load_cfg(
        self, file_path, fmt=None,
        preserve_case=True, **kwargs
    ):
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
            `"ini"`:
                `interpolation`: `None` to raw parse `%`.
                `strict`: `False` to allow non-safe files.
                `preserve_case`: `False` to lowercase option names.
        """
        fmt = io.get_file_format(file_path, fmt=fmt)
        if "json" in fmt:
            with open(file_path, "r") as f:
                d = json.load(f, **kwargs)
        elif "ini" in fmt:
            cp = configparser.ConfigParser(**kwargs)
            if preserve_case is True:
                cp.optionxform = str
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

    def get_items(self, use_all=False):
        """
        Gets configuration attributes.

        Returns
        -------
        sub_cfgs : `dict(str)`
            Keys of attributes which are configurations themselves.
        item_cfgs : `dict(str)`
            Configuration item keys.
        """
        if use_all:
            d = {k: v for k, v in self.__dict__.items()}
        else:
            d = self.attributes()
        sub_cfgs, item_cfgs = [], []
        for k, v in d.items():
            if isinstance(v, CfgBase):
                sub_cfgs.append(k)
            else:
                item_cfgs.append(k)
        return sub_cfgs, item_cfgs
