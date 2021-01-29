import copy
import numpy as np
import pandas as pd

from libics.core.data.arrays import ArrayData, SeriesData
from libics.core.util import misc


###############################################################################


class DataSequence(pd.DataFrame):

    """
    Stores a multiindex series as a pandas data frame.

    May contain arbitrary heterogeneous data, i.a. supports `ArrayData`.
    Typically this constructor does not have to be directly invoked, instead
    use the :py:func:`data.conversion.cv_list_to_datasequence` function
    to generate a `DataSequence` from a list of complex objects.
    For `pandas.DataFrame`-like usage of simple types, use the
    `data.arrays.SeriesData` class instead.

    Parameters
    ----------
    *args
        Arguments passed to the `pandas.DataFrame` constructor,
        particularly, includes the tabular data.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return DataSequence

    def rename_column(self, old_name, new_name):
        """
        Changes the column name.
        """
        self.rename({old_name: new_name}, inplace=True, axis="columns")

    def join(self, *args, how="outer", **kwargs):
        """
        Wrapper to :py:meth:`pd.DataFrame.join` with a different
        default behavior for parameter `how`.
        """
        return super().join(*args, how=how, **kwargs)

    def drop_column(self, *col_names, inplace=True):
        """
        Wrapper to :py:meth:`pd.DataFrame.drop(col=col_names)`.
        """
        return self.drop(columns=list(col_names), inplace=inplace)

    def apply_func(
        self, func, col_names, ret_name=True, drop_cols=False,
        print_progress=False
    ):
        """
        Applies a function to each row.

        Parameters
        ----------
        func : `callable`
            Call signature: `func(*col_names_values)->res`.
        col_names : `object` or `iter(object)`
            Column names whose values are passed to `func`.
        ret_name : `str` or `Iter[str]` or `bool`
            Column name in which return value is stored.
            If `Iter`, assumes that `func` returns an `Iter`. Each returned
            value is assigned to the respective column in order. If a certain
            value should not be assigned, pass `None` instead of `str`.
            If `True`, assigns return value to the row
            corresponding to the first item in `col_names`.
            If `False`, does not assign any return value.
        drop_cols : `bool`
            Flag whether to drop the columns used as `func` input.
        print_progress : `bool`
            Whether to print a progress bar.
        """
        if isinstance(col_names, str):
            col_names = [col_names]
        if ret_name is True:
            ret_name = col_names[0]
        args = self[list(col_names)]
        _iter = args.iterrows()
        if print_progress:
            _iter = misc.iter_progress(_iter)
        ret = [func(*arg) for _, arg in _iter]
        if ret_name is not False:
            if isinstance(ret_name, str):
                self[ret_name] = ret
            else:
                for i, name in enumerate(ret_name):
                    if name is not None:
                        self[name] = [item[i] for item in ret]
        if drop_cols:
            if ret_name in col_names:
                col_names.remove(ret_name)
            self.drop_column(*col_names)

    def sort_rows(self, col_name, ascending=True, **kwargs):
        """
        Sorts the rows by numerical values in a column.

        Parameters
        ----------
        col_name : `str`
            Column name of values to be sorted by.
        ascending : `bool`
            Flag whether to sort ascending (or descending).
        """
        self.sort_values(col_name, ascending=ascending, inplace=True, **kwargs)

    def reset_index(self, **kwargs):
        super().reset_index(drop=True, inplace=True, **kwargs)

    def average(self, key_columns, col_name, add_std=False, add_num=True):
        """
        Averages (unweighted mean) a column.

        Parameters
        ----------
        key_columns : `str` or `iter(str)` or `None`
            Columns which are used to distinguish different parameters,
            i.e. rows with the same values in the `key_columns` are
            averaged.
            If `None`, performs average on all rows.
        col_name : `str`
            Column to be averaged.
        add_std : `bool`
            Flag whether the standard deviation is added to the data sequence.
        add_num : `bool`
            Flag whether the averaging number is added to the data sequence.

        Returns
        -------
        dseq : `DataSequence`
            Averaged data sequence.

        Notes
        -----
        Uses the :py:func:`np.mean` and :py:func:`np.std` functions.
        """
        if key_columns is None:
            return self._average_all(
                col_name, add_std=add_std, add_num=add_num
            )
        if isinstance(key_columns, str):
            key_columns = [key_columns]
        # Find unique keys
        unique_keys = {}
        for idx, row in self.iterrows():
            key = tuple(row[_kc] for _kc in key_columns)
            if key not in unique_keys:
                unique_keys[key] = []
            unique_keys[key].append(row[col_name])
        # Construct DataSequence dict
        dseq_dict = {}
        _keys = tuple(unique_keys.keys())
        _key_array = np.transpose(np.array(_keys))
        for i, _kc in enumerate(key_columns):
            dseq_dict[_kc] = _key_array[i]
        # Perform average
        dseq_dict[col_name] = [
            self._calc_mean(unique_keys[k], axis=0) for k in _keys
        ]
        if add_std:
            dseq_dict["{:s}_std".format(col_name)] = [
                self._calc_std(unique_keys[k], axis=0) for k in _keys
            ]
        if add_num:
            dseq_dict["{:s}_num".format(col_name)] = [
                len(unique_keys[k]) for k in _keys
            ]
        # Construct DataSequence
        dseq = DataSequence(dseq_dict)
        return dseq

    def _average_all(self, col_name, add_std=False, add_num=True):
        dseq_dict = {}
        # Perform average
        dseq_dict[col_name] = self._calc_mean(self[col_name], axis=0)
        if add_std:
            dseq_dict["{:s}_std".format(col_name)] = self._calc_std(
                self[col_name], axis=0
            )
        if add_num:
            dseq_dict["{:s}_num".format(col_name)] = len(self[col_name])
        # Construct DataSequence
        dseq = DataSequence([dseq_dict])
        return dseq

    @staticmethod
    def _calc_mean(objs, *args, **kwargs):
        # Assumes that all objects have the same type
        objs0 = (
            objs.iloc[0] if isinstance(objs, (pd.DataFrame, pd.Series))
            else objs[0]
        )
        obj = None
        if isinstance(objs0, ArrayData) or isinstance(objs0, SeriesData):
            obj = copy.deepcopy(objs0)
            objs = [_obj.data for _obj in objs]
        if obj is None:
            obj = np.mean(objs, *args, **kwargs)
        else:
            obj.data = np.mean(objs, *args, **kwargs)
        return obj

    @staticmethod
    def _calc_std(objs, *args, **kwargs):
        # Assumes that all objects have the same type
        objs0 = (
            objs.iloc[0] if isinstance(objs, (pd.DataFrame, pd.Series))
            else objs[0]
        )
        obj = None
        if isinstance(objs0, ArrayData) or isinstance(objs0, SeriesData):
            obj = copy.deepcopy(objs0)
            objs = [_obj.data for _obj in objs]
        if obj is None:
            obj = np.std(objs, *args, **kwargs)
        else:
            obj.data = np.std(objs, *args, **kwargs)
        return obj
