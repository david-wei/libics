# System Imports
import math
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
# Identity Functions
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
