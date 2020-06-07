import pandas as pd


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
