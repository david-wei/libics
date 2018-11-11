import copy

import numpy as np
import pandas as pd

from libics import cfg
from libics.file import hdf


###############################################################################


class SeriesData(pd.DataFrame, hdf.HDFDelegate):

    """
    Stores a multiindex series as a pandas data frame. May contain arbitrary
    heterogeneous data, particularly also supports `ArrayData`.

    Examples
    --------
    >>> sd = SeriesData({
    ...     "cfg0": "cfg0-0 cfg0-1 cfg0-0 cfg0-1 cfg0-0 cfg0-1 cfg0-0".split(),
    ...     "cfg1": "cfg1-0 cfg1-1 cfg1-2 cfg1-0 cfg1-2 cfg1-2 cfg1-1".split(),
    ...     "stp0": np.arange(7),
    ...     "stp1": np.arange(7) * 2
    ... })
    >>> # Column slicing (variable slicing)
    >>> sd["cfg1"].T
                0      1      2      3      4      5      6
    cfg1   cfg1-0 cfg1-1 cfg1-2 cfg1-0 cfg1-2 cfg1-2 cfg1-1
    >>> sd[["cfg1", "stp1"]].T
               0      1      2      3      4      5      6
    cfg1  cfg1-0 cfg1-1 cfg1-2 cfg1-0 cfg1-2 cfg1-2 cfg1-1
    stp1       0      2      4      6      8     10     12
    >>> # Array slicing (same as numpy array slicing)
    >>> sd.loc[2]
    cfg0 cfg0-0
    cfg1 cfg1-2
    stp0      2
    stp1      4
    >>> sd.loc[:, ["cfg11", "stp1"]] == sd[["cfg1", "stp1"]]
    True
    >>> sd.loc[(sd["cfg0"] == "cfg0-0") & (sd["stp1"] >= 2), ["cfg0", "stp0"]]
          cfg0   stp0
    2   cfg0-0      2
    4   cfg0-0      4
    """

    def __init__(self, *args, cfg=None, stp=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg      # Data recording configuration
        self.stp = stp      # Measurement setup

    def __getitem__(self, key):
        try:
            super().__get_item__(key)
        except KeyError:
            # Exception might occur if column name is fulfilled implicitly
            # in self.cfg or self.stp
            if isinstance(key, tuple):
                if self.__has_row(key[0]) and self.__has_column(key[1]):
                    return self.__get_ext_selection(key[1])[key[0]]
                else:
                    raise
            # Row index exception is always exception
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
        elif column in self.cfg.__dict__.keys():
            return True
        elif column in self.stp.__dict__.keys():
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
                return _obj.assign({column: self[column]})
            elif column in self.cfg.__dict__.keys():
                return _obj.assign({
                    column: self.shape[0] * [self.cfg.__dict__[column]]
                })
            elif column in self.stp.__dict__.keys():
                return _obj.assign({
                    column: self.shape[0] * [self.stp.__dict__[column]]
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
        return _SeriesDataHDFDelegate(
            cfg_d=self.cfg.to_obj_dict(),
            stp_d=self.stp.to_obj_dict(),
            data_d=d
        )

    @property
    def _delegate_cls(self):
        return _SeriesDataHDFDelegate


class _TmpCfg(cfg.CfgBase):

    """FIXME: !!!!!!!!!!!"""

    def __init__(self, object_dict={}):
        super().__init__()
        self.__dict__.update(object_dict)

    def get_hl_cfg(self, *args, **kwargs):
        return self


_TmpStp = _TmpCfg       # FIXME: !!!!!!!!!!!


class _SeriesDataHDFDelegate(hdf.HDFBase):

    def __init__(self, cfg_d={}, stp_d={}, data_d={}):
        self.cfg_d = {}
        self.stp_d = {}
        self.data_d = {}

    def _from_delegate(self):
        # FIXME: construct cfg.get_hl_cfg (also for stp)
        return SeriesData(
            self.data_d,
            cfg=_TmpCfg(self.cfg_d).get_hl_cfg(),
            stp=_TmpStp(self.stp_d).get_hl_cfg()
        )


###############################################################################


def cv_list_to_seriesdata(ls):
    """
    Converts a data list into a `SeriesData` object.

    Parameters
    ----------
    ls : list
        List of scalars or structured data types. Particularly
        supports objects with configuration `cfg` and setup `stp`
        attributes. These can be analyzed in the SeriesData's
        underlying pandas.DataFrame structure.

    Returns
    -------
    sd : SeriesData or self
        Converted SeriesData object.
        If conversion is unsuccessful, returns input.
    """
    # 1. assert ls iterable
    try:
        iter(ls)
        assert(len(ls) > 0)
    except(TypeError, AssertionError):
        return ls
    # 2. assert items are compatible (scalar or arraydata with following)
    _cfg_type = None
    _has_cfg = np.all([hasattr(item, "cfg") for item in ls])
    if _has_cfg:
        _cfg_type = type(ls[0].cfg)
        _has_cfg = np.all([isinstance(item.cfg, _cfg_type) for item in ls])
    _stp_type = None
    _has_stp = np.all([hasattr(item, "stp") for item in ls])
    if _has_stp:
        _stp_type = type(ls[0].stp)
        _has_stp = np.all([isinstance(item.stp, _stp_type) for item in ls])
    # 3. filter varying and congruent configuration and settings items
    _col_cfg, _col_stp, _cfg, _stp = [], [], None, None
    if _has_cfg:
        _cfg = copy.deepcopy(ls[0].cfg)
        for key, val in _cfg.__dict__.items():
            if key != "kwargs" and key != "_msg_queue":
                for item in ls[1:]:
                    if getattr(item.cfg, key) != val:
                        _col_cfg.append(key)
                        break
    if _has_cfg:
        _stp = copy.deepcopy(ls[0].stp)
        for key, val in _stp.__dict__.items():
            if key != "kwargs" and key != "_msg_queue":
                for item in ls[1:]:
                    if getattr(item.stp, key) != val:
                        _col_stp.append(key)
                        break
    # 4. create dataframe from list of varying items and data itself
    _d = {"data": ls}
    for c in _col_cfg:
        _d[c] = [getattr(item.cfg, c) for item in ls]
    for c in _col_stp:
        _d[c] = [getattr(item.stp, c) for item in ls]
    return SeriesData(_d, cfg=_cfg, stp=_stp)


def cv_seriesdata_to_list(sd):
    """
    Retrieves the data stored in the `SeriesData` object.

    Parameters
    ----------
    sd : SeriesData
        SeriesData object storing the requested data.

    Returns
    -------
    ls : list
        Data list.
    """
    return list(sd.data)
