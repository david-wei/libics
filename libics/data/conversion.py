# System Imports
import numpy as np

# Package Imports
try:
    from . import addpath   # noqa
except(ImportError):
    import addpath          # noqa
import cfg.err as ERR

# Subpackage Imports
import data.matrixdata as matrixdata
import data.nddata as nddata


###############################################################################
# Data Type Conversion Functions
###############################################################################


def convert_MatrixData_to_StaticNdData(matrix_data):
    """
    Converts a MatrixData object into a StaticNdData object.

    Parameters
    ----------
    matrix_data : `data.matrixdata.MatrixData`
        The original data.

    Returns
    -------
    static_nddata : `data.nddata.StaticNdData`
        The converted data.

    Raises
    ------
    cfg.err.DTYPE_MATRIXDATA
        If parameters are invalid.
    """
    ERR.assertion(ERR.DTYPE_MATRIXDATA,
                  type(matrix_data) == matrixdata.MatrixData)
    static_nddata = nddata.StaticNdData()
    # Convert meta information
    static_nddata.set_pquants(matrix_data.get_pquants())
    static_nddata.init_data(len(matrix_data.get_data()))
    # Ravel data
    var = np.indices(matrix_data.get_data().shape).ravel()
    var_size = len(var) / 2
    val = matrix_data.get_data().ravel()
    # Calculate StaticNdData representation
    static_nddata.add_data(
        [(*matrix_data.get_pquant_by_index(var[i], var[i + var_size]), val[i])
         for i in range(var_size)],
        entry_index=0,
        data_form="list(tuple)"
    )
    return static_nddata
