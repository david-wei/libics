import math
import operator
import os
import re

import numpy as np


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
        return (data, )


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
    AssertionError
        If the parameters are invalid.
    """
    assert(type(string) == str and type(suffix) == str)
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


def flatten_nested_dict(d, delim="."):
    """
    Flattens a nested dictionary into a top-level dictionary. Levels are
    separated by the specified delimiter. Note that dictionary keys may
    only be of `str` type.

    Parameters
    ----------
    d : dict
        Dictionary to be flattened.
    delim : str
        Level delimiter for keys.

    Returns
    -------
    d : dict
        Flattened dictionary.
    """
    for key, val in list(d.items()):
        if isinstance(val, dict):
            for k, v in list(flatten_nested_dict(val, delim=delim).items()):
                d[key + delim + k] = v
            del d[key]
    return d


def nest_flattened_dict(d, delim="."):
    """
    Nests ("unflattens") a flattened dictionary. Levels are assumed to be
    separated by the specified delimiter.

    Parameters
    ----------
    d : dict
        Flattened dictionary.
    delim : str
        Level delimiter of keys.

    Returns
    -------
    d : dict
        Nested dictionary.
    """
    for key, val in list(d.items()):
        if delim in key:
            key_top, key_next = delim.split(key, 1)
            if key_top not in d.keys():
                d[key_top] = {}
            d[key_top][key_next] = val
            del d[key]
    for key, val in list(d.items()):
        if isinstance(val, dict):
            nest_flattened_dict(val, delim=delim)
    return d


def map_dicts(
    func, *arg_dicts, kwarg_dicts={},
    pre_args=(), post_args=(), **kwargs
):
    """
    Constructs a new dictionary containing all common keys. Calls the given
    function with the key's corresponding values and assigns the return value
    to the new dictionary.

    Parameters
    ----------
    func : callable
        Function to be mapped.
    *arg_dicts : dict
        Dictionaries whose values are mapped as arguments.
    kwarg_dicts : dict(dict)
        Dictionaries whose values are mapped as keyword arguments.
        The keys of kwarg_dicts correspond to the keywords.
    pre_args, post_args : tuple
        Arguments passed to func (before, after) dict
        values, i.e. func(*pre_args, *dict_vals, *post_args).
    **kwargs
        Keyword arguments passed to func.

    Returns
    -------
    d : dict
        Mapped dictionary.

    Example
    -------
    map_dicts(func, d1, d2) performs new_dict[key] = func(d1, d2) for all
    keys common to both d1 and d2.
    """
    d = {}
    all_dicts = list(arg_dicts) + list(kwarg_dicts.values())
    if len(all_dicts) == 1:
        kw = None
        if len(kwarg_dicts) == 1:
            kw = next(iter(kwarg_dicts.keys()))
        for key, val in all_dicts[0].items():
            if kw:
                kwd = {kw: val}
                d[key] = func(*pre_args, *post_args, **kwd, **kwargs)
            else:
                d[key] = func(*pre_args, val, *post_args, **kwargs)
    else:
        for key, val in all_dicts[0].items():
            vals = []
            kw_vals = {}
            is_common_key = True
            if is_common_key:
                for dd in arg_dicts:
                    if key in dd.keys():
                        vals.append(dd[key])
                    else:
                        is_common_key = False
                        break
            if is_common_key:
                for kw, dd in kwarg_dicts.items():
                    if key in dd.keys():
                        kw_vals[kw] = dd[key]
                    else:
                        is_common_key = False
                        break
            if is_common_key:
                d[key] = func(
                    *pre_args, *vals, *post_args, **kw_vals, **kwargs
                )
    return d


def make_dict(key_list, val_list):
    """
    Creates a dictionary from a key list and a value list.

    Parameters
    ----------
    key_list, val_list : list
        Same-length lists with keys and values.
        Mapping is performed on an index basis.

    Returns
    -------
    d : dict
        Created dictionary with key_list as keys and
        val_list as values.

    Raises
    ------
    AssertionError
        If the lengths of the lists do not match.
    """
    assert(len(key_list) == len(val_list))
    d = {}
    for i, k in enumerate(key_list):
        d[k] = val_list[i]
    return d


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


def split_strip(s, delim=",", strip=" \t\n\r"):
    """
    Splits a given string and strips each list element from given characters.

    Parameters
    ----------
    s : str
        String to be split.
    delim : str or None
        String delimiter. If None, string will not be split.
    strip : str
        Strip characters.

    Returns
    -------
    ls : list(str) or str
        Returns stripped (list of) string depending on delim
        parameter.
    """
    if delim is None:
        return s.strip(strip)
    else:
        ls = s.split(delim)
        return [item.strip(strip) for item in ls]


def split_unit(s):
    """
    Splits a given string into a tuple of numerical value and unit.

    Parameters
    ----------
    s : str
        String to be split.

    Returns
    -------
    val : int or float or str
        Numerical value or unchanged string.
    unit : str
        Unit.

    Notes
    -----
    If no numerical value could be determined, unit is None and the given
    string is returned as value.
    If only a numerical value could be determined, unit is None and the
    given string is cast to numerical format.
    """
    val, unit = None, None
    match = re.search(r"([+-]?\d*(?:\.\d+)?)(.*)", s)
    if match:
        val, unit = match.group(1), match.group(2)
        if val is None:
            val = unit
            unit = None
        else:
            try:
                val = int(val)
            except ValueError:
                val = float(val)
    else:
        val = s
    return val, unit


def extract(s, regex, group=1, cv_func=None, flags=0):
    """
    Extracts part of a string.
    Serves as convenience function to re.search.

    Parameters
    ----------
    s : str
        String from which to extract.
    regex : str
        Regular expression defining search function.
        Search findings should be enclosed in parentheses `()`.
    group : int or list(int) or tuple(int) or np.ndarray(1, int)
        Group index of search results.
        If list, returns corresponding list of search results.
    cv_func : callable or None
        Conversion function applied to search results (e.g. float).
    flags : int
        Flags parameter passed to re.search.

    Returns
    -------
    result
        Converted (if applicable) search result or results
        as defined by the `group` parameter.

    Raises
    ------
    KeyError
        If extraction failed.
    """
    match = re.search(regex, s, flags=flags)
    if match is None:
        raise KeyError("no match found")
    result = []
    _groups = group
    if np.isscalar(group):
        _groups = [group]
    _groups = list(_groups)
    for group_index in _groups:
        _extracted = match.group(group_index)
        if callable(cv_func):
            _extracted = cv_func(_extracted)
        result.append(_extracted)
    if np.isscalar(group):
        result = result[0]
    elif isinstance(group, tuple):
        result = tuple(result)
    elif isinstance(group, np.ndarray):
        result = np.array(result)
    return result


def capitalize_first_char(s):
    """
    Capitalizes the first character (if possible) and leaves the rest of the
    string invariant.

    Parameters
    ----------
    s : str
        String to be capitalized.

    Returns
    -------
    s_cap : str
        Capitalized string.
    """
    s_cap = re.sub("([a-zA-Z])", lambda x: x.groups()[0].upper(), s, 1)
    return s_cap


###############################################################################
# List Manipulations
###############################################################################


def order_list(ls, order):
    """
    Sorts a list according to a given index order.

    Parameters
    ----------
    ls : list
        List to be sorted.
    order : list(int)
        Sorting index order.

    Returns
    -------
    ls : list
        Sorted list.
    """
    return [ls[idx] for idx in order]


def flatten_nested_list(ls):
    """
    Flattens a nested list recursively.

    Parameters
    ----------
    ls : list
        Nested list.

    Returns
    -------
    flat_ls : list
        Flattened list.

    Notes
    -----
    Inefficient implementation.
    """
    flat_ls = []
    ls = assume_list(ls)
    for item in ls:
        if isinstance(item, list):
            flat_ls += flatten_nested_list(item)
        else:
            flat_ls += [item]
    return flat_ls


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
# Array Functions
###############################################################################


def resize_numpy_array(ar, shape, fill_value=0, mode_keep="front"):
    """
    Pads or shrinks a numpy array to a requested shape.

    Parameters
    ----------
    ar : numpy.ndarray
        Numpy array to be resized.
    shape : tuple(int)
        Target shape of array. Must have same dimension as ar.
    fill_value : ar.dtype
        Pad value if array is increased.
    mode_keep : "front", "back", "center"
        Keeps the elements of ar at the <mode_keep> of the
        resized array.

    Returns
    -------
    ar : numpy.ndarray
        Resized numpy array.
    """
    for dim in range(len(ar.shape)):
        start, stop = 0, shape[dim]
        if mode_keep == "front":
            start, stop = 0, shape[dim]
        elif mode_keep == "back":
            start, stop = ar.shape[dim] - shape[dim], ar.shape[dim]
        elif mode_keep == "center":
            diff = (ar.shape[dim] - shape[dim]) // 2
            start, stop = diff, shape[dim] + diff
        # reduce
        if shape[dim] < ar.shape[dim]:
            idx = [slice(None)] * dim + [slice(start, stop)]
            ar = ar[tuple(idx)]
        # expand
        elif shape[dim] > ar.shape[dim]:
            padl_shape = list(ar.shape)
            padl_shape[dim] = start
            padl = np.full(padl_shape, fill_value, dtype=ar.dtype)
            padr_shape = list(ar.shape)
            padr_shape[dim] = shape[dim] - stop
            padr = np.full(padr_shape, fill_value, dtype=ar.dtype)
            ar = np.concatenate((padl, ar, padr), axis=dim)
    return ar


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


def id_dict(arg):
    """
    Takes any iterable and returns a dictionary mapping the item values to
    themselves.
    """
    d = {}
    for item in arg:
        d[item] = item
    return d


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
