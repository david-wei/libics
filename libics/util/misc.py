# System Imports
import math
import operator
import os

# Package Imports
from libics.cfg import err as ERR


###############################################################################
# Assumption Functions
###############################################################################


def assume_iter(data):
    """
    Asserts that data is iterable. If the given data is iterable, it is
    returned. If it is not, a list with the data as single element is returned.

    Parameters
    ----------
    data
        Input data to be checked for iterability.

    Returns
    -------
    data : `iterable`
        Returns an iterable version of `data`.
    """
    try:
        iter(data)
    except(TypeError):
        data = [data]
    return data


def assume_tuple(data):
    """
    Asserts that data is a tuple. If the given data is a list, it is cast to a
    tuple. If not, a tuple is created with the data as single item.

    Parameters
    ----------
    data
        Input data to be checked for list.

    Returns
    -------
    data : tuple
        Returns a tuple containing `data`.
    """
    if isinstance(data, tuple):
        return data
    elif isinstance(data, list):
        return tuple(data)
    else:
        return tuple(data)


def assume_list(data):
    """
    Asserts that data is a list. If the given data is a tuple, it is cast to a
    list. If not, a list is created with the data as single item.

    Parameters
    ----------
    data
        Input data to be checked for list.

    Returns
    -------
    data : list
        Returns a list containing `data`.
    """
    if isinstance(data, list):
        return data
    elif isinstance(data, tuple):
        return list(data)
    else:
        return [data]


def assume_endswith(string, suffix):
    """
    Asserts that the passed `string` ends with with `suffix`. If it does not,
    the suffix is appended. The assumed result is returned.

    Parameters
    ----------
    string : `str`
        String to be checked.
    suffix : `str`
        Assumed end of string.

    Returns
    -------
    string : `str`
        String with assumed ending.

    Raises
    ------
    cfg.err.DTYPE_STR
        If the parameters are invalid.
    """
    ERR.assertion(ERR.DTYPE_STR,
                  type(string) == str,
                  type(suffix) == str)
    if not string.endswith(suffix):
        string += suffix
    return string


def assume_dir(path):
    """
    Asserts that the given path directory exists. If not, the path will be
    created. `path` is interpreted as file path if it does not end with `"/"`
    and has an extension `*.*`, otherwise it is interpreted as directory.

    Parameters
    ----------
    path : `str`
        Path of required directory or path to file in required
        directory.

    Returns
    -------
    dir_path : `str`
        If the directory did not exist, it was created. The
        directory path is returned.
    """
    path, ext = os.path.splitext(path)
    if ext != "":
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def assume_construct_obj(obj, cls_, raise_exception=None):
    """
    Asserts that the given object is of `cls_` class. If not and `obj` is a
    dictionary, an instance of the class is constructed with the dictionary
    as keyword arguments.

    Parameters
    ----------
    obj : object or dict
        Object or keyword argument dictionary.
    cls_ : class
        Class of instance to be constructed.
    raise_exception : Exception or None
        Specifies the behaviour if object construction fails.
        None:
            Returns the input object.
        Exception:
            Raises the given exception.

    Returns
    -------
    obj : cls_
        Instance of given class or object itself.

    Raises
    ------
    raise_exception
        Specified exception, if applicable.
    """
    if isinstance(obj, cls_):
        return obj
    elif isinstance(obj, dict):
        try:
            return cls_(**obj)
        except TypeError:
            if raise_exception is None:
                return obj
            else:
                raise raise_exception
    else:
        if raise_exception is None:
            return obj
        else:
            raise raise_exception


###############################################################################
# Dictionary Manipulations
###############################################################################


def reverse_dict(d):
    """
    Constructs a value-to-key dictionary from the given dictionary. Does not
    check for duplicates, so duplicate mapping will be random.

    Parameters
    ----------
    d : `dict`
        Dictionary to be reversed.

    Returns
    -------
    rev_d : `dict`
        Reversed dictionary.
    """
    return {val: key for key, val in d.items()}


###############################################################################
# String Functions
###############################################################################


def generate_fill_chars(s, fill_char=" "):
    """
    Generates fill characters with the same length as `s`.

    Parameters
    ----------
    s : `int` or object which implements `len`
        Object or length of object defining length of fill
        character string.
    fill_char : `str`
        String defining the fill character. Multi-character
        `fill_char` produces a series of fill characters which is
        cut to a length defined by `s`.

    Returns
    -------
    fs : `str`
        Fill string with length defined by `s`.
    """
    if type(s) != int:
        s = len(s)
    lfch = len(fill_char)
    rep = math.ceil(float(s) / lfch)
    fs = (rep * fill_char)[:s]
    return fs


###############################################################################
# List Fetcher
###############################################################################


def get_first_elem_iter(ls):
    """
    Gets the first element of a (nested) iterable.

    Parameters
    ----------
    ls : iterable
        Indexed iterable (list, tuple).

    Returns
    -------
    elem
        First element of the iterable.
    """
    try:
        return get_first_elem_iter(ls[0])
    except TypeError:
        return ls


def _gcrec(prev_comb, rem_ls):
    """
    Get combinations recursively.

    Parameters
    ----------
    prev_comb
        Previous combination.
    rem_ls
        Remaining list.
    """
    ls = []
    if len(rem_ls) > 1:
        for cur_val in rem_ls[0]:
            ls += _gcrec(prev_comb + [cur_val], rem_ls[1:])
    else:
        for cur_val in rem_ls[0]:
            ls.append(prev_comb + [cur_val])
    return ls


def get_combinations(ls):
    """
    Takes an iterable of iterables and returns a list with all possible
    mutual, ordered combinations.

    Parameters
    ----------
    ls : iterable
        Iterable from which combinations are constructed.

    Returns
    -------
    comb : list
        Combinations list.

    Example
    -------
    >>>> ls = [(1, 2), (5, ), (7, 8)]
    >>>> get_combinations(ls)
    [[1, 5, 7], [2, 5, 7], [1, 5, 8], [2, 5, 8]]
    """
    return _gcrec([], ls)


###############################################################################
# Identity Functions
###############################################################################


def do_nothing(*args, **kwargs):
    """
    Takes any arguments and keyword arguments and does nothing.
    """
    pass


def ret_id(arg):
    """
    Takes any argument and returns the input.
    """
    return arg


###############################################################################
# Operator Mapping
###############################################################################


operator_mapping = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "&": operator.and_,
    "^": operator.xor,
    "|": operator.or_,
    "**": operator.pow,
    "%": operator.mod,
    "in": operator.contains,
    "is": operator.is_,
    "is not": operator.is_not,
    "<<": operator.lshift,
    ">>": operator.rshift,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt
}
