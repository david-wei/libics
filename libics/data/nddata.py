# System Imports
import copy
import numpy as np

# Package Imports
from libics.cfg import err as ERR
from libics.util import serialization as ser

# Subpackage Imports
from libics.data import types


###############################################################################


class NdDataBase(ser.DictSerialization, object):

    """
    Base class for StaticNdData and DynamicNdData. Provides information about
    the physical quantities and units.
    """

    def __init__(self):
        super().__init__()
        self.pquants = []   # Physical quantities

    #   #### Getter ####################

    def get_dim(self):
        """
        Gets the dimension of the data according to pquants.

        Returns
        -------
        len_pquants : `int`
            Length of list of physical quantities.
        """
        return len(self.pquants)

    #   #### Setter ####################

    def append_pquant(self,
                      pquant):
        """
        Appends given physical quantity to the pquants list.

        Parameters
        ----------
        pquant : `data.types.pquant`
            Physical quantity. Must be unique in this pquants.

        Raises
        ------
        cfg.err.DTYPE_PQUANT
            If the parameters are invalid.
        cfg.err.INVAL_UNIQUE
            If the pquant is not unique.
        """
        ERR.assertion(ERR.DTYPE_PQUANT, type(pquant) == types.pquant)
        ERR.assertion(ERR.INVAL_UNIQUE, pquant not in self.pquants)
        self.pquants.append(pquant)

    def set_pquants(self,
                    pquants, overwrite=False):
        """
        Adds the given physical quantity to the pquants list. If the overwrite
        flag is set, any previously stored pquant will be deleted.

        Parameters
        ----------
        pquants : `[data.types.pquant]`
            Physical quantity.
        overwrite : `bool`
            If `True`, removes all stored physical quantities

        Raises
        ------
        cfg.err.DTYPE_STR, cfg.err.DTYPE_PQUANT
            If the parameters are invalid.
            If raised, any changes are reverted.
        cfg.err.INVAL_UNIQUE
            If any pquants item is not unique.
            If raised, any changes are reverted.
        """
        ERR.assertion(ERR.DTYPE_LIST, type(pquants) == list)
        pquants_old = None
        append_counter = 0
        try:
            # Maintain copy of old self.pquants
            if overwrite:
                pquants_old = copy.deepcopy(self.pquants)
                self.pquants = []
            # Append all passed pquants
            for pquant in pquants:
                self.append_pquant(pquant)
                append_counter += 1
        except(ERR.DTYPE_PQUANT, ERR.INVAL_UNIQUE):
            # Revert all changes
            for _ in range(append_counter):
                self.pquants.pop()
            self.pquants = pquants_old
            raise


###############################################################################


class StaticNdData(NdDataBase, object):

    """
    Stores multidimensional data in a 2D numpy ndarray. Functions as a
    static, performant data structure. All needed memory is supposed to be
    allocated at initialization. For dynamic data sizes, consider the
    `data.nddata.DynamicNdData` class.
    """

    def __init__(self):
        super().__init__()
        # Data structure (numpy.ndarray(1), dtype deducted from pquants)
        self.data = np.zeros(0)
        self.dtype = None
        # Counter flag indicating how many data entries are in use
        self.fill_count = 0
        # DictSerialization init
        self.add_obj_ser_func(
            {
                "data": [ser.numpy_ndarray_to_json, self.get_data]
            },
            {
                "data": [ser.json_to_numpy_ndarray]
            }
        )

    def init_data(self,
                  data_entry_count):
        """
        Initializes the data memory by considering the passed data entry count
        and the dtype attribute. Previously set pquants required.

        Parameters
        ----------
        data_entry_count : `int`
            Number of data points.

        Raises
        ------
        cfg.err.DTYPE_INT, cfg.err.INVAL_POS
            If the parameters are invalid.
        """
        ERR.assertion(ERR.DTYPE_INT, type(data_entry_count) == int)
        ERR.assertion(ERR.INVAL_POS, data_entry_count > 0)
        self.update_dtype()
        self.data = np.zeros(data_entry_count, dtype=self.dtype)
        self.fill_status = 0

    #   #### Getter ####################

    def get_data(self):
        """Gets `data` attribute."""
        return self.data

    #   #### Checks ####################

    def is_filled(self):
        """
        Checks if the allocated memory is filled according to a counter value.

        Returns
        -------
        filling_status : `bool`
            Flag whether data is filled.
        """
        return self.fill_count >= len(self.data)

    #   #### Setter ####################

    def update_dtype(self):
        """
        Gets the dtype for numpy structured array definition from pquants.
        Stores this str in the dtype attribute. The physical quantity name is
        used as numpy array column access.
        """
        self.dtype = [(pquant.name, pquant.dtype) for pquant in self.pquants]

    def add_data(self,
                 data_entries, entry_index=None, data_form="list(tuple)"):
        """
        Adds a data block to the data structure.

        Parameters
        ----------
        data_entries : `iterable(iterable)` or `numpy.ndarray(1)`
            Data block in a format as specified in data_form.
        entry_index : `int` or `None`, optional
            Data entry index at which the new block should be stored.
            Using `None`, the data will be stored after all filled
            entries.
        data_form : `str`
            `"list(list)"`: The data has an `iterable(iterable)` form
                and needs to be transformed to a list of tuples
                first.
            `"list(tuple)"`: The data has an `iterable(tuple)` form
                and can be directly transformed into a numpy
                structured array.
            `"numpy.structarray"`: The data is already in the right
                form and can be readily copied into the data
                structure.

        Raises
        ------
        AssertionError
            If the parameters are invalid. This includes falsely
            formatted data, insufficient memory or invalid data_form.
        """
        if entry_index is None:
            entry_index = self.fill_count
        max_new_entry_index = entry_index + len(data_entries)  # i.e. index + 1
        ERR.assertion(ERR.INDEX_BOUND, max_new_entry_index <= len(self.data))
        try:
            # If raw data entries are not tuples
            if data_form == "list(list)":
                for entry in data_entries:
                    self.data[entry_index] = (
                        self.convert_to_numpy_structured_array_via_tuple(entry)
                    )
                    entry_index += 1
            elif data_form == "list(tuple)":
                self.data[entry_index : max_new_entry_index] = (  # noqa (E203 bug)
                    np.array(data_entries, dtype=self.dtype)
                )
            elif data_form == "numpy.structarray":
                self.data[entry_index : max_new_entry_index] = (  # noqa (E203 bug)
                    data_entries
                )
            else:
                raise(AssertionError)
            # Update fill count
            self.fill_count = np.max((self.fill_count, max_new_entry_index))
        except(ValueError):
            raise(AssertionError)

    #   #### Calculator ################

    def convert_to_numpy_structured_array_via_tuple(self,
                                                    raw_data_entry):
        """
        Converts a data entry into a tuple (assuming an iterable data type),
        then converts it to a single item numpy structured array with dtype
        as specified by the dtype attribute.

        Returns
        -------
        data_entry : `np.ndarray(1)`
            Data entry in numpy structured array form.
        """
        return np.array(tuple(raw_data_entry), dtype=self.dtype)

    def sort_by_index(self, sorted_indices):
        """
        Sorts the data by a given index order list.

        Parameters
        ----------
        sorted_indices : `iterable(int)`
            List of sort order indices.

        Raises
        ------
        AssertionError
            If length of sorted_indices deviates from data length.
        """
        ERR.assertion(ERR.INDEX_SIZE, len(sorted_indices) == len(self.data))
        self.data[:] = self.data[sorted_indices]

    def sort_by_pquant(self, pquant_tuple):
        """
        Sorts the data according to a given pquant_tuple. The tuple consist of
        the structured array names and their order defines the sorting
        priority.

        Parameters
        ----------
        pquant_tuple : `iterable(str)`
            Tuple of structured array names ordered by sorting
            priority.

        Raises
        ------
        AssertionError
            If the parameters are invalid.
        """
        pquant_tuple = tuple(pquant_tuple)
        ERR.assertion(ERR.INVAL_STRUCT_NUMPY_NAME,
                      np.all([(pquant in self.data.dtype.names)
                              for pquant in pquant_tuple]))
        sorted_indices = self.data.argsort()
        self.sort_by_index(sorted_indices)
        return sorted_indices

    #   #### Operator ##################

    def __setitem__(self, key, value):
        """
        Parameters
        ----------
        key : `int` or `str`
            `int`: Addresses data entries.
            `str`: Addresses the respective physical quantity.
        value : `numpy.ndarray(1)`
            The numpy structured array item with dtype as defined in attribute.
        """
        self.data[key] = value

    def __getitem__(self, key):
        """
        Parameters
        ----------
        key : `int` or `str`
            `int`: Addresses data entries.
            `str`: Addresses the respective physical quantity.

        Returns
        -------
        value : `numpy.ndarray(1)`
            Returns the data entry or physical quantity array that
            corresponds to the passed key.
        """
        return self.data[key]


###############################################################################


class DynamicNdData(NdDataBase, object):

    pass
