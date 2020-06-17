import pandas as pd

from libics.core.data.types import Quantity
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
        self._quantity = {}

    @property
    def _constructor(self):
        return DataSequence

    def set_quantity(self, **kwargs):
        for k, v in kwargs.items():
            self._quantity[k] = misc.assume_construct_obj(v, Quantity)

    def get_quantity(self):
        return self._quantity

    @property
    def quantity(self):
        q = self._quantity.copy()
        for col in self.columns:
            if col not in q:
                q[col] = Quantity(name=col)
        return q

    @staticmethod
    def append(*args, **kwargs):
        """
        Wrapper for :py:meth:`pandas.append`.
        """
        return pd.append(*args, **kwargs)
