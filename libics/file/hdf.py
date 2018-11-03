import copy
import inspect
import json

import h5py
import numpy as np

from libics.cfg import env as ENV


# FIXME: Implement self-construction of objects from _hdf_cls_name

###############################################################################


class HDFBase(object):

    """
    Base class for HDF file serialization specifying serialization parameters.

    For saving a custom object to HDF5 files, inherit from this base class
    and pass the identifier names (package reference and class name) to init.
    Make sure that all custom class attributes are also derived from this base
    class.

    Parameters
    ----------
    pkg_name : str
        Package name of serialized class.
    cls_name : str
        Class name of serialized class.

    Notes
    -----
    Supported data types (for automatic serialization):
    * numpy-castable types
    * unstructured nested lists and tuples with numpy castable
      type list elements
    """

    HDFBase_ATTRS_LEN = 2

    def __init__(self, pkg_name="libics", cls_name="HDFBase"):
        self._hdf_pkg_name = pkg_name
        self._hdf_cls_name = cls_name


class HDFList(HDFBase):

    """
    Conversion class to allow non-numpy-like lists and tuples to be HDF
    serialized.

    Example
    -------
    >>> # Serialization procedure
    >>> ls = [1, 2, [3, 4], "bla"]
    >>> hdf_ls = HDFList()
    >>> hdf_ls.from_list(ls)
    >>> hdf_ls.__dict__
    {"it0": 1, "it1": 2, "it2": [3, 4], "it3": "bla"}
    >>> hdf_ls.to_list() == ls
    True
    """

    def __init__(self):
        super().__init__(pkg_name="libics", cls_name="HDFList")

    def from_list(self, ls):
        """
        Loads a given list into the object's attributes with the list index
        as key.

        Parameters
        ----------
        ls : list or tuple
            List to be loaded into object dictionary.
        """
        for it, val in enumerate(ls):
            self.__dict__["it" + str(it)] = val

    def to_list(self):
        """
        Constructs a list from the ascendingly named object dictionary.

        Returns
        -------
        ls : list
            Reconstructed list.
        """
        ls = []
        counter = 0
        while True:
            str_counter = "it" + str(counter)
            if str(str_counter) in self.__dict__.keys():
                ls.append(self.__dict__[str_counter])
                counter += 1
            else:
                break
        return ls


class HDFDict(HDFBase):

    """
    Conversion class to allow built-in dictionaries to be HDF serialized.

    Converts the dictionary to an HDFDict object that has the dictionary
    key-value pairs as attributes.

    Notes
    -----
    * This indirect HDF serialization allows coherent procedures for all
      classes.
    * Only strings are allowed as keys.
    """

    def __init__(self):
        super().__init__(pkg_name="libics", cls_name="HDFDict")

    def from_dict(self, d):
        """
        Loads a given dictionary into the object's attributes.

        Parameters
        ----------
        d : dict
            Dictionary to be loaded into object dictionary.
        """
        self.__dict__.update(d)

    def to_dict(self):
        """
        Constructs a dictionary from the object dictionary. Removes
        serialization metadata elements.

        Returns
        -------
        d : dict
            Reconstructed dictionary.
        """
        d = copy.deepcopy(self.__dict__)
        del d["_hdf_pkg_name"]
        del d["_hdf_cls_name"]
        return d


###############################################################################


def write_hdf(obj, file_path=None, _parent_group=None):
    """
    Writes a given object into an HDF5 file.

    Parameters
    ----------
    obj : HDFBase
        The object to be written. Needs to be derived from
        the `HDFBase` class which specifies serialization
        procedures.
    file_path : str or None
        The file path of the HDF5 file to be written.
        Used only for top level file creation.
        If `None`, writes to the given `_parent_group`.
    _parent_group : h5py.Group
        HDF5 parent group used for recursive file writing.
        If `file_path` is specified, this parameter is
        ignored.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    TypeError
        If the given `obj` has attributes not derived from
        the `HDFBase` class.
    """
    # Create top level group (HDF file)
    if file_path is not None:
        with h5py.File(file_path, "w") as file_group:
            file_group.attrs["LIBICS_VERSION"] = ENV.LIBICS_VERSION
            write_hdf(obj, file_path=None, _parent_group=file_group)
    # Write child data
    else:
        for key, val in obj.__dict__.items():
            # Custom implemented class
            if isinstance(val, HDFBase):
                write_hdf(
                    val, file_path=None,
                    _parent_group=_parent_group.create_group(key)
                )
            # Binary numpy data
            elif type(val) == np.ndarray:
                _parent_group.create_dataset(key, data=val)
            # Python dictionary
            elif type(val) == dict:
                dict_obj = HDFDict()
                dict_obj.from_dict(val)
                write_hdf(
                    dict_obj, file_path=None,
                    _parent_group=_parent_group.create_group(key)
                )
            # Unstructured Python list
            elif type(val) == list or type(val) == tuple:
                # Check if numpy castable
                is_numpy_castable = True
                ar = None
                try:
                    if type(val) == list:
                        is_numpy_castable = False
                    else:
                        ar = np.array(val)
                        if ar.dtype == "O":
                            is_numpy_castable = False
                except ValueError:
                    is_numpy_castable = False
                # Numpy-castable
                if is_numpy_castable:
                    # Unicode string
                    if "U" in str(ar.dtype):
                        vlds = _parent_group.create_dataset(
                            key, ar.shape, dtype=h5py.special_dtype(vlen=str)
                        )
                        for count, item in enumerate(ar):
                            vlds[count] = item
                    # Other types
                    else:
                        _parent_group.create_dataset(key, data=ar)
                # Unstructured list
                else:
                    list_obj = HDFList()
                    list_obj.from_list(val)
                    write_hdf(
                        list_obj, file_path=None,
                        _parent_group=_parent_group.create_group(key)
                    )
            # Simple (built-in) data type
            else:
                _parent_group.attrs[key] = val


def read_hdf(cls_or_obj, file_path=None,
             _parent_group=None, _ignore_undeclared_attrs=True):
    """
    Reads an HDF5 file into an object.

    Parameters
    ----------
    cls_or_obj : class or object
        The object to which the read data is stored.
        If a class is given, a default object of it is created.
    file_path : str or None
        The file path of the HDF5 file to be read from.
        If `None`, reads from the given `_parent_group`.
    _parent_group : h5py.Group
        HDF5 parent group used for recursive file writing.
        If `file_path` is specified, this parameter is
        ignored.
    _ignore_undeclared_attrs : bool
        Whether to read attributes that have not been specified
        in the class definition.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    TypeError
        If the given `obj` has attributes not derived from
        the `HDFBase` class.

    Notes
    -----
    * Object attributes which are saved in the HDF5 file but not
      declared in the given object are silently ignored if the
      `_ignore_undeclared_attrs` flag is set.
      Otherwise the respective attributes are copied.
    * Attributes of HDF5 data sets are silently ignored.
    """
    # Check if class or object
    if inspect.isclass(cls_or_obj):
        cls_or_obj = cls_or_obj()
    # Open file
    if file_path is not None:
        with h5py.File(file_path, "r") as file_group:
            read_hdf(cls_or_obj, file_path=None, _parent_group=file_group)
    # Process child data
    else:
        # Simple (built-in) data types
        for key, val in _parent_group.attrs.items():
            if (
                not _ignore_undeclared_attrs
                or key in cls_or_obj.__dict__.keys()
            ):
                cls_or_obj.__dict__[key] = val
        # Groups or data sets
        for key, val in _parent_group.items():
            if (
                not _ignore_undeclared_attrs
                or key in cls_or_obj.__dict__.keys()
            ):
                # Data sets
                if isinstance(val, h5py.Dataset):
                    cls_or_obj.__dict__[key] = val[()]
                # Group
                elif isinstance(val, h5py.Group):
                    # Unstructured list
                    if val.attrs["_hdf_cls_name"] == "HDFList":
                        hdf_ls = HDFList()
                        hdf_ls.from_list([None] * (
                            len(val) + len(val.attrs)
                            - HDFBase.HDFBase_ATTRS_LEN
                        ))
                        hdf_ls = read_hdf(
                            hdf_ls, file_path=None, _parent_group=val
                        )
                        cls_or_obj.__dict__[key] = hdf_ls.to_list()
                    # Python dictionary
                    elif val.attrs["_hdf_cls_name"] == "HDFDict":
                        hdf_d = read_hdf(
                            HDFDict, file_path=None, _parent_group=val,
                            _ignore_undeclared_attrs=False
                        )
                        cls_or_obj.__dict__[key] = hdf_d.to_dict()
                    # Custom objects
                    else:
                        cls_or_obj.__dict__[key] = read_hdf(
                            cls_or_obj.__dict__[key], file_path=None,
                            _parent_group=val
                        )
    # Return reconstructed object
    return cls_or_obj


###############################################################################


def _to_dict(obj):
    """
    Serializes a given object into a dictionary.

    Parameters
    ----------
    obj : object
        Object to be serialized.

    Returns
    -------
    d : dict or self
        Dictionary serialized object.
    """
    # Check if obj is an object
    if not hasattr(obj, "__dict__"):
        return obj
    d = {}
    for key, val in obj.__dict__.items():
        # HDFBase object
        if isinstance(val, HDFBase):
            d[key] = _to_dict(val)
        # Python list
        elif isinstance(val, list) or isinstance(val, tuple):
            ls = list(val[:])
            for i, item in enumerate(val):
                ls[i] = _to_dict(item)
            d[key] = ls + ["_hdf_is_list"]
        # Python dict
        elif isinstance(val, dict):
            d[key] = val
            d[key]["_hdf_is_dict"] = True
        # Numpy array
        elif isinstance(val, np.ndarray):
            d[key] = val.tolist()
        # Simple types
        else:
            d[key] = val
    return d


def _from_dict(obj, d):
    """
    Deserializes a dictionary into a given object.

    Parameters
    ----------
    obj : object
        Object to be reconstructed.
    d : dict
        Dictionary to be deserialized.

    Returns
    -------
    obj : object
        Reconstructed object.
    """
    for key, val in d.items():
        # Dictionary
        if isinstance(val, dict):
            # HDFBase object
            if "_hdf_cls_name" in val.keys():
                obj.__dict__[key] = _from_dict(obj.__dict__[key], val)
            # Python dict
            else:
                obj.__dict__[key] = val
                del obj.__dict__[key]["_hdf_is_dict"]
        # List
        elif isinstance(val, list):
            # Python list
            if val[-1] == "_hdf_is_list":
                obj.__dict__[key] = val[:-1]
            # Numpy array
            else:
                obj.__dict__[key] = np.array(val)
        # Simple types
        else:
            obj.__dict__[key] = val
    return obj


def write_json(obj, file_path, indent=2):
    """
    Writes a given object into a json file.

    Parameters
    ----------
    obj : object
        The object to be written. Needs to be json serializable.
        This includes built-in types, `numpy.ndarray` and classes
        derived from the `HDFBase` class.
    file_path : str
        The file path of the json file to be written.
    indent : int or None
        JSON pretty-printing indent spaces. None uses
        no newlines.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    TypeError
        If the given `obj` has attributes not derived from
        the `HDFBase` class.
    """
    json.dump(_to_dict(obj), open(file_path, "w"), indent=indent)


def read_json(cls_or_obj, file_path):
    """
    Reads a json file into an object.

    Parameters
    ----------
    cls_or_obj : class or object
        The object to which the read data is stored.
        If a class is given, a default object of it is created.
    file_path : str
        The file path of the json file to be read from.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if inspect.isclass(cls_or_obj):
        cls_or_obj = cls_or_obj()
    return _from_dict(cls_or_obj, json.load(open(file_path, "r")))


###############################################################################


if __name__ == "__main__":

    # Test HDFList
    ls = [1, 2, [3, 4], "bla"]
    print("List:", ls)
    hdf_ls = HDFList()
    hdf_ls.from_list(ls)
    print("HDFList:", hdf_ls.__dict__)
    print("Reconstruction test:", hdf_ls.to_list() == ls)
    print("")

    # Test class configuration
    import os
    file_name = "file_hdf_test"
    tmp_dir = os.path.join(ENV.DIR_PKGROOT, "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    file_path = os.path.join(tmp_dir, file_name)

    class SubTest(HDFBase):

        def __init__(self):
            super().__init__(pkg_name="libics", cls_name="SubTest")
            self.y_list = [1, 2, 3, 4, 5, 6, 7]

        def reset(self):
            self.y_list = []

    class Test(HDFBase):

        def __init__(self):
            super().__init__(pkg_name="libics", cls_name="Test")
            self.x_str = "Hello, world!"
            self.x_int = 123456789
            self.x_float = 3.14159
            self.x_tuple_str = ("bli", "bla", "blub°°")
            self.x_nestedlist_float = [1.23, 4.56, [7.89, 0.12], 3.45]
            self.x_dict_mixed = {"a": 0, "b": "b", "3": "c", "4": 4, "5.0": -5}
            self.x_subtest = SubTest()

        def reset(self):
            self.x_str = ""
            self.x_int = 0
            self.x_float = 0.0
            self.x_tuple_str = ("",)
            self.x_nestedlist_float = [1.0, [2.0]]
            self.x_dict_mixed = {"reset": "reset"}
            self.x_subtest.reset()

        def __str__(self):
            s = "x_str: " + self.x_str + "\n"
            s += "x_int: " + str(self.x_int) + "\n"
            s += "x_float: " + str(self.x_float) + "\n"
            s += "x_tuple_str: " + str(self.x_tuple_str) + "\n"
            s += "x_nestedlist_float: " + str(self.x_nestedlist_float) + "\n"
            s += "x_dict_mixed: " + str(self.x_dict_mixed) + "\n"
            s += "x_subtest.y_list: " + str(self.x_subtest.y_list)
            return s

    # Test write_hdf
    x = Test()
    print("--------------\n x (to write)\n--------------")
    print(str(x))
    print("")
    write_hdf(x, file_path=(file_path + ".hdf5"))

    # Test read_hdf
    y = Test()
    y.reset()
    print("--------------\n y (reset)\n--------------")
    print(str(y))
    print("")
    read_hdf(y, file_path=(file_path + ".hdf5"))
    print("--------------\n y (read)\n--------------")
    print(str(y))
    print("")

    # Test write_json
    z = Test()
    print("--------------\n z (to write)\n--------------")
    print(str(z))
    print("")
    write_json(z, file_path + ".json")

    # Test read_json
    u = Test()
    u.reset()
    print("--------------\n u (reset)\n--------------")
    print(str(u))
    print("")
    read_json(u, file_path + ".json")
    print("--------------\n u (read)\n--------------")
    print(str(u))
    print("")
