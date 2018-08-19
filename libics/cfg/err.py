###############################################################################
# LibICS Exception Framework
###############################################################################


class ErrorDescription(object):

    """
    Provides exception descriptions.

    Each custom exception should have an instance of `ErrorDescription` as
    class variable. It contains a default description of the exception.

    Parameters
    ----------
    description : str
        Default exception description.
    code : positive int or None, optional
        Numerical error code.

    Raises
    ------
    TypeError
        If a passed data type is invalid.
    """

    def __init__(self, description, code=None):
        if type(description) != str:
            raise TypeError("invalid description ({:s})"
                            .format(str(type(description))))
        if type(code) is not None and (type(code) != int or code <= 0):
            raise TypeError("invalid code ({:s})"
                            .format(str(type(code))))
        self.description = description
        self.code = code

    def __str__(self):
        """
        Generates a string of error code (if available) and error description.
        """
        s = ""
        if self.code is not None:
            s = "[{:d}] ".format(self.code)
        s += self.description
        return s

    def __int__(self):
        """
        Gets the (positive) error code.

        For a missing error code, `-1` is returned.
        """
        code = self.code
        if code is None:
            code = -1
        return code


class LibICSError(Exception):

    """
    Generic LibICS exception base class.

    Implements `ErrorDescription` class object and an error description class
    method.
    """

    _err_description = ErrorDescription(
        "generic LibICS error",
        code=1
    )

    @classmethod
    def str(cls):
        return str(cls._err_description)


def assertion(exception, *args, description=None):
        """
        Asserts that the passed values are `True`.

        Parameters
        ----------
        exception : Exception or inherited
            Exception to be raised
        *args : bool
            Boolean values to be checked for validity.
        description : str
            Exception description that overwrites the default
            description.

        Raises
        ------
        CustomException
            If any `args` value is not `True`, the exception that
            subclassed this class is raised.
        """
        if not issubclass(exception, Exception):
            raise Exception("invalid type: non-exception type")
        for arg in args:
            if type(arg) != bool or arg is False:
                if description is None and issubclass(exception, LibICSError):
                    raise exception(exception.str())
                else:
                    raise exception(description)


###############################################################################
# LibICS Exception Implementation
###############################################################################


# ++++++++++++++++++++++++++++++++++++++++++++++++++
# 100: Data Type
# ++++++++++++++++++++++++++++++++++++++++++++++++++


class DTYPE(LibICSError):
    _err_description = ErrorDescription(
        "data type: generic error",
        100
    )


class DTYPE_SCALAR(DTYPE):
    _err_description = ErrorDescription(
        "data type: expected scalar",
        110
    )


class DTYPE_BOOL(DTYPE_SCALAR):
    _err_description = ErrorDescription(
        "data type: expected bool",
        111
    )


class DTYPE_INT(DTYPE_SCALAR):
    _err_description = ErrorDescription(
        "data type: expected int",
        112
    )


class DTYPE_FLOAT(DTYPE_SCALAR):
    _err_description = ErrorDescription(
        "data type: expected float",
        113
    )


class DTYPE_COMPLEX(DTYPE_SCALAR):
    _err_description = ErrorDescription(
        "data type: expected complex",
        114
    )


class DTYPE_ITER(DTYPE):
    _err_description = ErrorDescription(
        "data type: expected iterable",
        120
    )


class DTYPE_STR(DTYPE_ITER):
    _err_description = ErrorDescription(
        "data type: expected str",
        121
    )


class DTYPE_TUPLE(DTYPE_ITER):
    _err_description = ErrorDescription(
        "data type: expected tuple",
        122
    )


class DTYPE_LIST(DTYPE_ITER):
    _err_description = ErrorDescription(
        "data type: expected list",
        123
    )


class DTYPE_DICT(DTYPE_ITER):
    _err_description = ErrorDescription(
        "data type: expected dict",
        124
    )


class DTYPE_NDARRAY(DTYPE_ITER):
    _err_description = ErrorDescription(
        "data type: expected numpy.ndarray",
        126
    )


class DTYPE_CALL(DTYPE):
    _err_description = ErrorDescription(
        "data type: expected callable",
        140
    )


class DTYPE_FUNC(DTYPE_CALL):
    _err_description = ErrorDescription(
        "data type: expected function",
        141
    )


class DTYPE_CUSTOM(DTYPE):
    _err_description = ErrorDescription(
        "data type: expected LibICS custom type",
        150
    )


class DTYPE_PQUANT(DTYPE_CUSTOM):
    _err_description = ErrorDescription(
        "data type: expected data.types.pquant",
        151
    )


class DTYPE_MATRIXDATA(DTYPE_CUSTOM):
    _err_description = ErrorDescription(
        "data type: expected data.matrixdata.MatrixData",
        151
    )


# ++++++++++++++++++++++++++++++++++++++++++++++++++
# 200: Index/Key Validity
# ++++++++++++++++++++++++++++++++++++++++++++++++++


class INDEX(LibICSError):
    _err_description = ErrorDescription(
        "index: generic error",
        200
    )


class INDEX_RANGE(INDEX):
    _err_description = ErrorDescription(
        "index: generic range error",
        210
    )


class INDEX_SIZE(INDEX_RANGE):
    _err_description = ErrorDescription(
        "index: invalid range",
        211
    )


class INDEX_BOUND(INDEX_RANGE):
    _err_description = ErrorDescription(
        "index: out of bounds error",
        212
    )


class INDEX_INDINV(INDEX_RANGE):
    _err_description = ErrorDescription(
        "index: invalid index",
        213
    )


class INDEX_KEYINV(INDEX_RANGE):
    _err_description = ErrorDescription(
        "index: invalid key",
        214
    )


class INDEX_DIM(INDEX_RANGE):
    _err_description = ErrorDescription(
        "index: dimension error",
        215
    )


# ++++++++++++++++++++++++++++++++++++++++++++++++++
# 300: Value Validity
# ++++++++++++++++++++++++++++++++++++++++++++++++++


class INVAL(LibICSError):
    _err_description = ErrorDescription(
        "invalid value: generic error",
        300
    )


class INVAL_SIGN(INVAL):
    _err_description = ErrorDescription(
        "invalid value: generic sign error",
        310
    )


class INVAL_NONZERO(INVAL_SIGN):
    _err_description = ErrorDescription(
        "invalid value: expected non-zero value",
        311
    )


class INVAL_POS(INVAL_SIGN):
    _err_description = ErrorDescription(
        "invalid value: expected positive value",
        312
    )


class INVAL_NEG(INVAL_SIGN):
    _err_description = ErrorDescription(
        "invalid value: expected negative value",
        313
    )


class INVAL_ZERO(INVAL_SIGN):
    _err_description = ErrorDescription(
        "invalid value: expected zero value",
        314
    )


class INVAL_NONPOS(INVAL_SIGN):
    _err_description = ErrorDescription(
        "invalid value: expected non-positive value",
        315
    )


class INVAL_NONNEG(INVAL_SIGN):
    _err_description = ErrorDescription(
        "invalid value: expected non-negative value",
        316
    )


class INVAL_SET(INVAL):
    _err_description = ErrorDescription(
        "invalid value: expected value from defined set",
        320
    )


class INVAL_NONEMPTY(INVAL_SET):
    _err_description = ErrorDescription(
        "invalid value: expected non-empty value",
        321
    )


class INVAL_UNIQUE(INVAL_SET):
    _err_description = ErrorDescription(
        "invalid value: expected unique value from defined set",
        322
    )


class INVAL_PARAM(INVAL):
    _err_description = ErrorDescription(
        "invalid value: generic function parameters error",
        330
    )


class INVAL_STRUCT(INVAL):
    _err_description = ErrorDescription(
        "invalid value: generic structured data error",
        340
    )


class INVAL_STRUCT_NUMPY(INVAL_STRUCT):
    _err_description = ErrorDescription(
        "invalid value: generic numpy data error",
        341
    )


class INVAL_STRUCT_NUMPY_NAME(INVAL_STRUCT_NUMPY):
    _err_description = ErrorDescription(
        "invalid value: numpy custom data type name error",
        3411
    )


class INVAL_STRUCT_WCT(INVAL_STRUCT):
    _err_description = ErrorDescription(
        "invalid value: generic WCT file error",
        342
    )
