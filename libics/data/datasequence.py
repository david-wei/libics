import copy
import os
import uuid

import numpy as np
import pandas as pd

from libics.data import stp, types
from libics.drv import drv
from libics.file import hdf
from libics.util import misc
from libics import env


###############################################################################


class DataSequence(pd.DataFrame, hdf.HDFDelegate):

    """
    Stores a multiindex series as a pandas data frame.

    May contain arbitrary heterogeneous data, i.a. supports `ArrayData`.
    Typically this constructor does not have to be directly invoked, instead
    use the `cv_list_to_datasequence` function to generate a `DataSequence`
    from a list of complex objects. For `pandas.DataFrame`-like usage of
    simple types, use the `SeriesData` class instead.

    Parameters
    ----------
    *args
        Arguments passed to the `pandas.DataFrame` constructor,
        particularly, includes the tabular data.
    cfg : dict
        Flattened dictionary of driver configuration.
    stp : dict
        Flattened dictionary of setup configuration.

    Examples
    --------
    >>> ds = DataSequence({
    ...     "cfg0": "cfg0-0 cfg0-1 cfg0-0 cfg0-1 cfg0-0 cfg0-1 cfg0-0".split(),
    ...     "cfg1": "cfg1-0 cfg1-1 cfg1-2 cfg1-0 cfg1-2 cfg1-2 cfg1-1".split(),
    ...     "stp0": np.arange(7),
    ...     "stp1": np.arange(7) * 2
    ... })
    >>> # Column slicing (variable slicing)
    >>> ds["cfg1"].T
                0      1      2      3      4      5      6
    cfg1   cfg1-0 cfg1-1 cfg1-2 cfg1-0 cfg1-2 cfg1-2 cfg1-1
    >>> ds[["cfg1", "stp1"]].T
               0      1      2      3      4      5      6
    cfg1  cfg1-0 cfg1-1 cfg1-2 cfg1-0 cfg1-2 cfg1-2 cfg1-1
    stp1       0      2      4      6      8     10     12
    >>> # Array slicing (same as numpy array slicing)
    >>> ds.loc[2]
    cfg0 cfg0-0
    cfg1 cfg1-2
    stp0      2
    stp1      4
    >>> ds.loc[:, ["cfg11", "stp1"]] == ds[["cfg1", "stp1"]]
    True
    >>> ds.loc[(ds["cfg0"] == "cfg0-0") & (ds["stp1"] >= 2), ["cfg0", "stp0"]]
          cfg0   stp0
    2   cfg0-0      2
    4   cfg0-0      4
    """

    _metadata = ["cfg", "stp"]

    def __init__(self, *args, cfg=None, stp=None, quantity=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg      # Data recording configuration
        self.stp = stp      # Measurement setup
        self.quantity = quantity

    @property
    def _constructor(self):
        return DataSequence

    @property
    def quantity(self):
        for col in self.columns:
            if col not in self._quantity.keys():
                self._quantity[col] = types.Quantity(name=col)
        return self._quantity

    @quantity.setter
    def quantity(self, val):
        self._quantity = {} if quantity is None else quantity

    def __getitem__(self, key):
        """
        Parameters
        ----------
        key : tuple
            key[0]: row -> single.
            key[1]: column -> single.
        key : single (scalar, slice, list)
            Column, as str (column name).
        """
        try:
            return super().__getitem__(key)
        except(KeyError, AttributeError):
            # Exception might occur if column name is fulfilled implicitly
            # in self.cfg or self.stp
            if isinstance(key, tuple):
                if self.__has_row(key[0]) and self.__has_column(key[1]):
                    return self.__get_ext_selection(key[1])[key[0]]
                else:
                    raise
            # Column name exception
            elif self.__has_column(key):
                return self.__get_ext_selection(key)
            # Other exception
            else:
                raise

    def __has_row(self, row):
        if isinstance(row, slice):
            return self.__has_row(row.start) and self.__has_row(row.stop)
        elif isinstance(row, list):
            for item in row:
                if not self.__has_row(item):
                    return False
            return True
        else:
            if abs(row) >= self.shape[0]:
                return False
            else:
                return True

    def __has_column(self, column):
        if isinstance(column, slice):
            return False
        elif isinstance(column, list):
            for item in column:
                if not self.__has_column(item):
                    return False
            return True
        elif column in self.columns:
            return True
        elif column in self.cfg.keys():
            return True
        elif column in self.stp.keys():
            return True
        else:
            return False

    def __get_ext_selection(self, column, _obj=None):
        if _obj is None:
            _obj = pd.DataFrame()
        if isinstance(column, list):
            for item in column:
                _obj = (self.__get_ext_selection(item, _obj=_obj))
            return _obj
        else:
            if column in self.columns:
                return _obj.assign(**{column: self[column]})
            elif column in self.cfg.keys():
                return _obj.assign(**{
                    column: self.shape[0] * [self.cfg[column]]
                })
            elif column in self.stp.keys():
                return _obj.assign(**{
                    column: self.shape[0] * [self.stp[column]]
                })

    def _to_delegate(self):
        d = {}
        length = self.shape[0]
        for c in self.columns:
            _same_type = True
            if length > 1:
                for it in range(length - 1):
                    if (
                        not isinstance(c[it], type(c[it + 1]))
                        or hasattr(c[it], "__dict__")
                    ):
                        _same_type = False
                        break
            if _same_type:
                d[c] = np.array(self[c])
            else:
                d[c] = list(self[c])
        return _DataSequenceHDFDelegate(
            cfg_d=self.cfg, stp_d=self.stp, data_d=d
        )

    @property
    def _delegate_cls(self):
        return _DataSequenceHDFDelegate

    def get_names(self):
        """
        Gets the names/headers of the data sequence.

        Returns
        -------
        names : `dict(str->dict/set)`
            "cfg" -> cfg attribute dict.
            "stp" -> stp attribute dict.
            "data" -> data frame column names as set (without cfg, stp)
        """
        cfg_names = {} if self.cfg is None else self.cfg.__dict__
        stp_names = {} if self.stp is None else self.stp.__dict__
        for k in ["_hdf_pkg_name", "_hdf_cls_name", "_hdf_is_delegate",
                  "group", "_msg_queue"]:
            cfg_names.pop(k, None)
            stp_names.pop(k, None)
        data_names = set(self.columns).difference(
            set(cfg_names), set(stp_names)
        )
        return {"cfg": cfg_names, "stp": stp_names, "data": data_names}


@hdf.InheritMap(map_key=("libics", "_DataSequenceHDFDelegate"))
class _DataSequenceHDFDelegate(hdf.HDFBase):

    def __init__(self, cfg_d={}, stp_d={}, data_d={}):
        super().__init__(cls_name="_DataSequenceHDFDelegate")
        self.cfg_d = cfg_d
        self.stp_d = stp_d
        self.data_d = data_d

    def _from_delegate(self):
        return DataSequence(
            self.data_d, cfg=self.cfg_d, stp=self.stp_d
        )


###############################################################################


def cv_list_to_datasequence(ls):
    """
    Converts a data list into a `DataSequence` object.

    Parameters
    ----------
    ls : list
        List of scalars or structured data types. Particularly
        supports objects with configuration `cfg` and setup `stp`
        attributes. These can be analyzed in the DataSequence's
        underlying pandas.DataFrame structure.

    Returns
    -------
    ds : DataSequence or self
        Converted DataSequence object.
        If conversion is unsuccessful, returns input.

    Notes
    -----
    TODO: does not set quantity
    """
    # 1. assert ls iterable
    try:
        iter(ls)
        assert(len(ls) > 0)
    except(TypeError, AssertionError):
        return ls
    # 2. assert items are compatible (scalar or arraydata with following)
    _cfg_type = None
    _cfg_dicts = []
    _has_cfg = np.all([hasattr(item, "cfg") for item in ls])
    if _has_cfg:
        _cfg_type = type(ls[0].cfg)
        _has_cfg = np.all([isinstance(item.cfg, _cfg_type) for item in ls])
        if _has_cfg:
            _cfg_dicts = [misc.flatten_nested_dict(item.cfg.to_obj_dict())
                          for item in ls]
    _stp_type = None
    _stp_dicts = []
    _has_stp = np.all([hasattr(item, "stp") for item in ls])
    if _has_stp:
        _stp_type = type(ls[0].stp)
        _has_stp = np.all([isinstance(item.stp, _stp_type) for item in ls])
        if _has_stp:
            _stp_dicts = [misc.flatten_nested_dict(item.stp.to_obj_dict())
                          for item in ls]
    # 3. filter varying and congruent configuration and settings items
    _col_cfg, _col_stp, _cfg, _stp = [], [], None, None
    if _has_cfg:
        # construct reference dict containing all keys
        _cfg = copy.deepcopy(_cfg_dicts[0])
        for d in _cfg_dicts:
            _cfg.update(d)
        # determine varying items
        del_keys = []
        for key, val in _cfg.items():
            for item in _cfg_dicts:
                if key not in item.keys() or item[key] != val:
                    _col_cfg.append(key)
                    del_keys.append(key)
                    break
        for key in del_keys:
            del _cfg[key]
    if _has_stp:
        # construct reference dict containing all keys
        _stp = copy.deepcopy(_stp_dicts[0])
        for d in _stp_dicts:
            _stp.update(d)
        # determine varying items
        del_keys = []
        for key, val in _stp.items():
            for item in _stp_dicts:
                if key not in item.keys() or item[key] != val:
                    _col_stp.append(key)
                    break
        for key in del_keys:
            del _stp[key]
    # 4. create dataframe from list of varying items and data itself
    _d = {"data": ls}
    for c in _col_cfg:
        _d[c] = [item[c] if c in item.keys() else None for item in _cfg_dicts]
    for c in _col_stp:
        _d[c] = [item[c] if c in item.keys() else None for item in _stp_dicts]
    return DataSequence(_d, cfg=_cfg, stp=_stp)


def cv_datasequence_to_list(ds):
    """
    Retrieves the data stored in the `DataSequence` object.

    Parameters
    ----------
    ds : DataSequence
        DataSequence object storing the requested data.

    Returns
    -------
    ls : list
        Data list.
    """
    return list(ds.data)


def concatenate_datasequences(dss):
    """
    Concatenates a list of `DataSequence` objects to form a single large
    `DataSequence` object.

    Parameters
    ----------
    dss : `iter(DataSequence)`
        List of data sequences to be concatenated.

    Returns
    -------
    ds : `DataSequence`
        Concatenated data sequence.
    """
    misc.assume_iter(dss)
    # Find parameter names
    col_names, cfg_names, stp_names = set(), set(), set()
    cfg_dict, stp_dict = {}, {}
    cfg_init, stp_init = False, False
    for ds in dss:
        names = ds.get_names()
        cfg_names.update(names["cfg"].keys())
        if cfg_init:
            cfg_dict = dict(cfg_dict.items() & names["cfg"].items())
        else:
            cfg_dict.update(names["cfg"])
        stp_names.update(names["stp"].keys())
        if stp_init:
            stp_dict = dict(stp_dict.items() & names["stp"].items())
        else:
            stp_dict.update(names["stp"])
        col_names.update(names["data"])
    col_names.update(cfg_names - set(cfg_dict.keys()),
                     stp_names - set(stp_dict.keys()))
    # TODO: DataSequency.quantity in concatenate_datasequences,
    # TODO: cv_list_to_datasequence,
    # TODO: icsproj.expanlys.files.load_sequence_to_datasequence
    # Construct data sequence
    ds = pd.concat(dss)
    return ds


###############################################################################


@hdf.InheritMap(map_key=("libics", "DataSequenceFileMap"))
class DataSequenceFileMap(hdf.HDFBase):

    """
    Intermediate class for multi-file storage of DataSequence.

    Parameters
    ----------
    ds : DataSequence
        DataSequence object to be written.
    file_path : str
        File path of the DataSequence file.
    data_dir : str or None
        Directory where data is saved. If the data items have
        a non-`None` `file_path` attribute, this parameter is
        ignored.
        `None` chooses the directory of the DataSequence file.
    force_dir : bool
        Flag whether the data should be saved in `data_dir`
        regardless of its `file_path` attribute.
        Creates random file names.
    """

    def __init__(self,
                 ds=None, file_path=None, data_dir=None, force_dir=True):
        self.ds = ds
        self.file_path = file_path
        self.data_dir = data_dir
        self.force_dir = force_dir
        self.data_file_paths = None

    def write(self):
        """
        Writes each item from the data column to an HDF5 file and saves a JSON
        file map listing each data item's relative location.
        """
        if self.data_dir is None:
            self.data_dir = os.path.dirname(self.file_path)
        file_paths = []
        for d in self.ds.data:
            if not hasattr(d, "file_path") or getattr(d, "file_path") is None:
                file_paths.append(os.path.join(
                    self.data_dir, str(uuid.uuid1()) + ".hdf5"
                ))
            elif self.force_dir:
                file_paths.append(os.path.join(
                    self.data_dir, os.path.basename(getattr(d, "file_path"))
                ))
            else:
                file_paths.append(getattr(d, "file_path"))
            hdf.write_hdf(d, file_path=file_paths[-1])
        self.ds = None
        self.data_dir = os.path.relpath(self.data_dir, start=self.file_path)
        self.data_file_paths = [
            os.path.relpath(fp, start=os.path.dirname(self.file_path))
            for fp in file_paths
        ]
        hdf.write_json(self, self.file_path, indent=env.FORMAT_JSON_INDENT)

    def read(self, data_cls, file_path=None):
        """
        Reads a DataSequence from a JSON file map.

        Parameters
        ----------
        data_cls : class
            Class of the data items.
        file_path : str
            File path of the file map.

        Returns
        -------
        ds : DataSequence
            The DataSequence reconstructed from the file map.
        """
        if file_path is not None:
            self.file_path = file_path
        hdf.read_json(self, self.file_path)
        self.ds = [hdf.read_hdf(
            data_cls,
            os.path.join(os.path.dirname(self.file_path), fp)
        ) for fp in self.data_file_paths]
        return cv_list_to_datasequence(self.ds)
