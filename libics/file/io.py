import datetime
from importlib import import_module
import numpy as np
import pandas as pd
import json
import bson


from libics import env
from libics.util import misc
from libics.file import hdf


###############################################################################


def get_class_from_fqname(fqname):
    """
    Gets the class specified by the given fully qualified class name.

    Parameters
    ----------
    fqname : `str`
        Fully qualified class name (i.e. Python-attribute-style string
        specifying class), e.g. `libics.data.arraydata.ArrayData`.

    Returns
    -------
    _cls : `class`
        Requested class object.
    """
    module_path, _, class_name = fqname.rpartition('.')
    try:
        mod = import_module(module_path)
        _cls = getattr(mod, class_name)
        return _cls
    except (AttributeError, ValueError):
        raise ImportError("invalid class ({:s})".format(str(fqname)))


def get_fqname_from_class(_cls):
    """
    Gets the fully qualified class name from the given class.

    Parameters
    ----------
    _cls : `class`
        Class object.

    Returns
    -------
    fqname : `str`
        Requested fully qualified class name.

    Notes
    -----
    See: https://stackoverflow.com/questions/2020014
    """
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

    Raises
    ------
    NotImplementedError
        If class of object to be serialized is not supported.
    """

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
        if type_is_primitive(obj):
            return obj
        elif isinstance(obj, complex):
            return [ID_COMPLEX, obj.real, obj.imag]
        elif isinstance(obj, (list, tuple)):
            return [cls.serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: v for k, v in obj.items()}
        else:
            d = {}
            d["__cls__"] = get_fqname_from_class(type(obj))
            if isinstance(obj, FileBase):
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

    Raises
    ------
    KeyError
        If deserialization failed.
    """

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
                            obj = get_class_from_fqname(name)()
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
        return cls.deserialize(ser["__data__"], obj=obj, raise_err=raise_err)


###############################################################################


FILE_FORMATS = ["json", "bson", "hdf"]


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
    if fmt not in FILE_FORMATS:
        raise KeyError("invalid file format ({:s})".format(str(fmt)))
    return fmt


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
    ser = enc.encode(obj)
    if "json" in fmt:
        json.dump(ser, open(file_path, "w"), **kwargs)
    elif "bson" in fmt:
        stream = bson.BSON.encode(ser)
        with open(file_path, "wb") as f:
            f.write(stream)
    elif "hdf" in fmt:
        hdf.write_hdf(obj, file_path=file_path)
    else:
        raise NotImplementedError("format {:s} not supported".format(fmt))


def load(
    file_path, obj_or_cls=None, dec=ObjDecoder, fmt=None,
    req_version=None, raise_err=True
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
        File format: `"json"`, `"bson"`, `"hdf"`.
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
    """
    fmt = get_file_format(file_path, fmt=fmt)
    obj = obj_or_cls
    if isinstance(obj_or_cls, type):
        obj = obj_or_cls()
    if "json" in fmt:
        ser = json.load(open(file_path, "r"))
        obj = dec.decode(
            ser, obj=obj, req_version=req_version, raise_err=raise_err
        )
    elif "bson" in fmt:
        with open(file_path, "rb") as f:
            stream = f.read()
        ser = bson.BSON.decode(stream)
        obj = dec.decode(
            ser, obj=obj, req_version=req_version, raise_err=raise_err
        )
    elif "hdf" in fmt:
        obj = hdf.read_hdf(obj_or_cls, file_path=file_path)
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
      serialized. These attributes are given in the class variable `KEYS`,
      which is a set containing the respective attribute names.
    * The subclass should add the required attribute names to this set, i.e.
      setting `KEYS = FileBase.KEYS | {"ATTR0", "ATTR1", ...}`.
    * Alternatively, the :py:meth:`attributes` method itself can be
      overwritten to obtain more customizability.
    """

    KEYS = set()

    def __init__(self):
        raise NotImplementedError

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
        return {k: getattr(self, k) for k in self.KEYS}
