import datetime
import json
import numpy as np
import pandas as pd
import scipy.io
from importlib import import_module


from libics import env
from libics.core.io import image
from libics.core.util import misc

LOGGER = env.logging.get_logger("libics.core.io")

try:
    import bson
except ImportError:
    LOGGER.info(
        "Could not load `bson` package. "
        + "If you are accessing `*.bson` files, "
        + "install the Python package `pymongo`."
    )


###############################################################################


def get_class_from_fqname(fqname, cls_map=None):
    """
    Gets the class specified by the given fully qualified class name.

    Parameters
    ----------
    fqname : `str`
        Fully qualified class name (i.e. Python-attribute-style string
        specifying class), e.g. `libics.core.data.arrays.ArrayData`.
    cls_map : `dict(str->cls)`
        Dictionary mapping fully qualified name to a class.
        Is handled prioritized, so it can be used for legacy purposes,
        where dependencies have changed.

    Returns
    -------
    _cls : `class`
        Requested class object.
    """
    # Use class map
    if cls_map is not None and fqname in cls_map:
        return cls_map[fqname]
    # Use modules
    module_path, _, class_name = fqname.rpartition('.')
    try:
        mod = import_module(module_path)
        _cls = getattr(mod, class_name)
        return _cls
    except (AttributeError, ValueError):
        raise ImportError("invalid class ({:s})".format(str(fqname)))


def get_fqname_from_class(_cls, cls_map=None):
    """
    Gets the fully qualified class name from the given class.

    Parameters
    ----------
    _cls : `class`
        Class object.
    cls_map : `dict(cls->str)`
        Dictionary mapping class to a fully qualified name.
        Is handled prioritized, so it can be used for legacy purposes,
        where dependencies have changed.

    Returns
    -------
    fqname : `str`
        Requested fully qualified class name.

    Notes
    -----
    See: https://stackoverflow.com/questions/2020014
    """
    # Use class map
    if cls_map is not None and _cls in cls_map:
        return cls_map[_cls]
    # Use modules
    fqname = []
    module = _cls.__module__
    if not (module is None or module == str.__class__.__module__):
        fqname.append(module)
    fqname.append(_cls.__name__)
    return ".".join(fqname)


###############################################################################


def type_is_primitive(obj):
    """
    Returns whether the given object is of primitive type.

    Notes
    -----
    `complex` is not considered a primitive, but a special type.
    """
    return obj is None or type(obj) in (bool, int, float, str)


def filter_primitive(obj):
    """
    Filters primitive types requiring special handling (i.a. `np.generic`)
    into Python built-in types.

    Parameters
    ----------
    obj : `object`
        Object to be filtered.

    Returns
    -------
    obj : `object`
        Converted object if applicable, otherwise returns identity.
    """
    if isinstance(obj, np.generic):
        obj = obj.item()
    return obj


# Efficiently encode special types as lists with leading string `"_^ID"`
ID_COMPLEX = "_^CX"
ALL_IDS = (ID_COMPLEX,)


# Unique object as return value
class ID_NOT_SPECIAL:
    pass


class ObjEncoder(object):

    """
    Serializes an object to a file or dictionary.

    For serialization to file (with metadata), use :py:meth:`encode`.
    For legacy class to fully qualified name map (:py:attr:`CLS_MAP`),
    inherit from this class and populate the attribute according to
    :py:func:`get_fqname_from_class`.

    Raises
    ------
    NotImplementedError
        If class of object to be serialized is not supported.
    """

    CLS_MAP = None

    @classmethod
    def _serialize_numpy_ndarray(cls, obj):
        dtype = misc.get_numpy_dtype_str(obj.dtype)
        if dtype == "object":
            data = cls.serialize(obj.tolist())
        elif dtype == "complex":
            data = [
                ID_COMPLEX,
                np.real(obj).tolist(),
                np.imag(obj).tolist(),
            ]
        else:
            data = obj.tolist()
        d = {
            "dtype": dtype,
            "data": data,
        }
        return d

    @classmethod
    def _serialize_pandas_dataframe(cls, obj):
        np_data = cls._serialize_numpy_ndarray(obj.to_numpy())
        d = {
            "columns": list(obj.columns),
            "dtype": np_data["dtype"],
            "data": np_data["data"],
        }
        return d

    @classmethod
    def serialize(cls, obj):
        obj = filter_primitive(obj)
        if type_is_primitive(obj):
            return obj
        elif isinstance(obj, complex):
            return [ID_COMPLEX, obj.real, obj.imag]
        elif isinstance(obj, (list, tuple)):
            return [cls.serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: cls.serialize(v) for k, v in obj.items()}
        else:
            d = {}
            d["__cls__"] = get_fqname_from_class(
                type(obj), cls_map=cls.CLS_MAP
            )
            if isinstance(obj, FileBase) or hasattr(obj, "__LIBICS_IO__"):
                d["__obj__"] = cls.serialize(obj.attributes())
            elif isinstance(obj, np.ndarray):
                d["__obj__"] = cls._serialize_numpy_ndarray(obj)
            elif isinstance(obj, pd.DataFrame):
                d["__obj__"] = cls._serialize_pandas_dataframe(obj)
            elif "__dict__" in obj:
                d["__obj__"] = cls.serialize(obj.__dict__)
            else:
                raise NotImplementedError("invalid object ({:s})"
                                          .format(repr(obj)))
            return d

    @classmethod
    def encode(cls, obj):
        """
        Encodes an object into a serializable format.

        Parameters
        ----------
        obj : `object`
            Object to be deserialized.

        returns
        -------
        ser : `dict`
            Object serialized into a dictionary.
        """
        ser = {}
        now = datetime.datetime.now()
        ser["__meta__"] = {
            "version_libics": env.LIBICS_VERSION,
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
        }
        ser["__data__"] = cls.serialize(obj)
        return ser


class ObjDecoder(object):

    """
    Deserializes a file or dictionary to an object.

    For deserialization from file (with metadata), use :py:meth:`decode`.
    For legacy fully qualified name to class map (:py:attr:`CLS_MAP`),
    inherit from this class and populate the attribute according to
    :py:func:`get_class_from_fqname`.

    Raises
    ------
    KeyError
        If deserialization failed.
    """

    CLS_MAP = None

    @classmethod
    def _deserialize_special_types(cls, ser):
        if ser[0] == ID_COMPLEX:
            if len(ser) >= 3:
                return float(ser[1]) + 1j * float(ser[2])
        return ID_NOT_SPECIAL

    @classmethod
    def _deserialize_numpy_ndarray(cls, ser, raise_err=True):
        dtype = ser["dtype"]
        if dtype == "object":
            return np.array(cls.deserialize(ser["data"]), dtype=dtype)
        elif dtype == "complex":
            if ser["data"][0] == ID_COMPLEX and len(ser["data"]) == 3:
                return (
                    np.array(ser["data"][1], dtype=float)
                    + 1j * np.array(ser["data"][2], dtype=float)
                )
            elif raise_err:
                raise KeyError("invalid serialization format ({:s})"
                               .format("complex numpy.ndarray"))
            else:
                return ser["data"]
        else:
            return np.array(ser["data"], dtype=dtype)

    @classmethod
    def _deserialize_pandas_dataframe(cls, ser, raise_err=True):
        columns = ser["columns"]
        data = cls._deserialize_numpy_ndarray(ser, raise_err=raise_err)
        return pd.DataFrame(data=data, columns=columns)

    @classmethod
    def deserialize(cls, ser, obj=None, raise_err=True):
        """
        Parameters
        ----------
        ser : `dict`
            Dictionary representing serialized object.
        obj : `object`
            If specified, and if the top-level object is a custom class,
            it is deserialized into this given object.
        raise_err : `bool`
            Flag whether to raise an error on deserialization failure
            or to proceed silently.

        Returns
        -------
        obj : `object`
            Deserialized object.

        Raises
        ------
        KeyError
            If deserialization failed and `raise_err` flag is set.
        """
        # Dict-like
        if isinstance(ser, dict):
            # Class object
            if "__cls__" in ser:
                name = ser["__cls__"]
                # Assert serialization format
                if "__obj__" not in ser:
                    if raise_err:
                        raise KeyError("invalid class object ({:s})"
                                       .format(str(name)))
                    else:
                        return None
                data = ser["__obj__"]
                # Numpy.ndarray
                if name == "numpy.ndarray":
                    obj = cls._deserialize_numpy_ndarray(
                        data, raise_err=raise_err
                    )
                # Pandas.DataFrame
                elif name == "pandas.DataFrame":
                    obj = cls._deserialize_pandas_dataframe(
                        data, raise_err=raise_err
                    )
                # FileBase
                else:
                    # Construct object and fill attributes
                    try:
                        if obj is None:
                            obj = get_class_from_fqname(
                                name, cls_map=cls.CLS_MAP
                            )()
                        for k, v in data.items():
                            setattr(obj, k, cls.deserialize(v))
                    # If object construction fails
                    except ImportError:
                        if raise_err:
                            raise KeyError("invalid class name ({:s})"
                                           .format(str(name)))
                        else:
                            return data
            # Dict
            else:
                obj = {k: cls.deserialize(v) for k, v in ser.items()}
            return obj
        # Primitive
        elif type_is_primitive(ser):
            return ser
        # Iterable
        else:
            ser = misc.assume_iter(ser)
            # Empty list
            if len(ser) == 0:
                return ser
            # Special types (ID encoded)
            id_ser = ser[0]
            if isinstance(id_ser, str):
                ret = cls._deserialize_special_types(ser)
                if ret != ID_NOT_SPECIAL:
                    return ret
                else:
                    return ser
            # List or tuple
            else:
                return [cls.deserialize(item) for item in ser]

    @classmethod
    def decode(cls, ser, obj=None, req_version=None, raise_err=True):
        """
        Decodes a dictionary into a deserialized object.

        Parameters
        ----------
        ser : `dict`
            File-serialized object.
        obj : `object`
            If specified, and if the top-level object is a custom class,
            it is deserialized into this given object.
        req_version : `None`, `float` or `(float, float)`
            Required `libics` version. Tuples specify minimum and maximum
            versions, a single float a minimum version.
            Raises a `RuntimeError` if this condition is not fulfilled.
        raise_err : `bool`
            Flag whether to raise an error on deserialization failure
            or to proceed silently.

        Returns
        -------
        obj : `object`
            Deserialized object.

        Raises
        ------
        RuntimeError
            If the required version is not fulfilled.
        KeyError
            If the serialized object is invalid.
        """
        # Check version
        if req_version is not None:
            if np.isscalar(req_version):
                req_version = (req_version, None)
            version = misc.extract(
                ser["__meta__"]["libics_version"], r"(\d+.\d+).*", func=float
            )
            if (
                (req_version[0] is not None and req_version[0] > version)
                or (req_version[1] is not None and req_version[1] < version)
            ):
                raise RuntimeError("incompatible libics version ({:s})"
                                   .format(ser["__meta__"]["libics_version"]))
        # Deserialize data
        data = None
        if "__data__" in ser:   # libics encoded data
            data = ser["__data__"]
        else:
            data = ser          # undefined dict data
        return cls.deserialize(data, obj=obj, raise_err=raise_err)


###############################################################################


def get_file_format(file_path, fmt=None):
    """
    Deduces the file format (or returns the format if given).

    Raises
    ------
    KeyError
        If format deduction error occured.
    """
    if fmt is None:
        try:
            fmt = file_path.split(".")[-1]
        except IndexError:
            raise KeyError("could not deduce file format ({:s})"
                           .format(str(file_path)))
    fmt = fmt.lower()
    return fmt


_SAVER_FUNC = {}


def _register_save_fmt(fmt, func):
    """
    Registers `func` as saver function for `fmt` files in :py:func:`save`.

    Parameters
    ----------
    fmt : `str`
        File format.
    func : `callable`
        Saver function.
        Call signature: `func(file_path)->obj`.
    """
    _SAVER_FUNC[fmt] = func


def save(
    file_path, obj, enc=ObjEncoder, fmt=None, **kwargs
):
    """
    Serializes and saves an object to file.

    Parameters
    ----------
    file_path : `str`
        Saved file path.
    obj : `object`
        Object to be serialized and saved.
    enc : `ObjEncoder`
        Serialization class.
    fmt : `str`
        File format: `"json"`, `"bson"`, `"hdf"`.
    **kwargs
        Keyword arguments passed to the respective write functions.
    """
    fmt = get_file_format(file_path, fmt=fmt)
    # Registered fmt
    for k, v in _SAVER_FUNC.items():
        if k in fmt:
            return v(file_path)
    # Default fmt
    if "json" in fmt:
        ser = enc.encode(obj)
        with open(file_path, "w") as f:
            json.dump(ser, f, **kwargs)
    elif "bson" in fmt:
        ser = enc.encode(obj)
        stream = bson.BSON.encode(ser)
        with open(file_path, "wb") as f:
            f.write(stream)
    elif "csv" in fmt:
        ser = pd.DataFrame(obj)
        ser.to_csv(file_path, **kwargs)
    elif "xls" in fmt:
        ser = pd.DataFrame(obj)
        ser.to_excel(file_path, **kwargs)
    elif "hdf" in fmt:
        raise NotImplementedError("hdf format not yet implemented")
    else:
        raise NotImplementedError("format {:s} not supported".format(fmt))


_LOADER_FUNC = {}


def _register_load_fmt(fmt, func):
    """
    Registers `func` as loader function for `fmt` files in :py:func:`load`.

    Parameters
    ----------
    fmt : `str`
        File format.
    func : `callable`
        Loader function.
        Call signature: `func(file_path)->obj`.
    """
    _LOADER_FUNC[fmt] = func


def load(
    file_path, obj_or_cls=None, dec=ObjDecoder, fmt=None,
    req_version=None, raise_err=True, **kwargs
):
    """
    Loads and deserializes an object from file.

    Parameters
    ----------
    file_path : `str`
        Saved file path.
    obj_or_cls : `object`
        Object or class to be deserialized to.
    dec : `ObjDecoder`
        Deserialization class.
    fmt : `str`
        Supported file formats: `"json"`, `"bson"`, `"hdf"`,
        `"bmp"`, `"png"`, `"wct"`, `"sif"`, `"mat"`.
    req_version : `None`, `float` or `(float, float)`
        Required `libics` version. Tuples specify minimum and maximum
        versions, a single float a minimum version.
        Raises a `RuntimeError` if this condition is not fulfilled.
    raise_err : `bool`
        Flag whether to raise an error on deserialization failure
        or to proceed silently.
    **kwargs
        Keyword arguments passed to the loader function.

    Returns
    -------
    obj : `object`
        Deserialized object.
    """
    fmt = get_file_format(file_path, fmt=fmt)
    obj = obj_or_cls
    if isinstance(obj_or_cls, type):
        obj = obj_or_cls()
    # Registered fmt
    for k, v in _LOADER_FUNC.items():
        if k in fmt:
            return v(file_path)
    # Default fmt
    if "json" in fmt:
        with open(file_path, "r") as f:
            ser = json.load(f, **kwargs)
        obj = dec.decode(
            ser, obj=obj, req_version=req_version, raise_err=raise_err
        )
    elif "bson" in fmt:
        with open(file_path, "rb") as f:
            stream = f.read()
        ser = bson.BSON.decode(stream, **kwargs)
        obj = dec.decode(
            ser, obj=obj, req_version=req_version, raise_err=raise_err
        )
    elif "tif" in fmt or "tiff" in fmt:
        obj = image.load_tif_to_arraydata(file_path, ad=obj_or_cls, **kwargs)
    elif "bmp" in fmt:
        obj = image.load_bmp_to_arraydata(file_path, ad=obj_or_cls, **kwargs)
    elif "png" in fmt:
        obj = image.load_png_to_arraydata(file_path, ad=obj_or_cls, **kwargs)
    elif "wct" in fmt:
        obj = image.load_wct_to_arraydata(file_path, ad=obj_or_cls, **kwargs)
    elif "sif" in fmt:
        obj = image.load_sif_to_arraydata(file_path, ad=obj_or_cls, **kwargs)
    elif "mat" in fmt:
        from libics.core.data.types import AttrDict
        obj = AttrDict(scipy.io.loadmat(file_path))
    elif "csv" in fmt:
        obj = pd.read_csv(file_path, **kwargs)
    elif "xls" in fmt:
        obj = pd.read_excel(file_path, **kwargs)
    elif "hdf" in fmt:
        raise NotImplementedError("hdf format not yet implemented")
    else:
        raise NotImplementedError("format {:s} not supported".format(fmt))
    return obj


class FileBase(object):

    """
    Base class for object serialization and file storage.

    Usage:

    * To enable I/O functionality for custom classes, subclass this
      base class.
    * By default the serialization algorithm calls :py:meth:`attributes`,
      which returns a name->value dictionary of the attributes to be
      serialized. These attributes are given in the class variable `SER_KEYS`,
      which is a set containing the respective attribute names.
    * The subclass should add the required attribute names to this set, i.e.
      setting `SER_KEYS = FileBase.SER_KEYS | {"ATTR0", "ATTR1", ...}`.
    * Alternatively, the :py:meth:`attributes` method itself can be
      overwritten to obtain more customizability.
    """

    SER_KEYS = set()

    def save(self, file_path, **kwargs):
        """
        Wrapper for :py:func:`save` function.

        Saves current object.
        """
        save(file_path, self, **kwargs)

    def load(self, file_path, **kwargs):
        """
        Wrapper for :py:func:`load` function.

        Loads into current object.
        """
        return load(file_path, obj_or_cls=self, **kwargs)

    def attributes(self):
        """
        Default saved attributes getter.

        Returns
        -------
        attrs : `dict(str->object)`
            Saved attributes dictionary mapping the attribute name
            to the attribute value.
        """
        return {k: getattr(self, k) for k in self.SER_KEYS}
