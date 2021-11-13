import math
import numbers
import operator
import os
import re
import time

import numpy as np


###############################################################################
# Type Checks and Conversions
###############################################################################


def is_number(var):
    """
    Returns whether given variable is of scalar, numeric type.
    """
    return isinstance(var, numbers.Number)


def cv_float(text, dec_sep=[".", ","]):
    """
    Converts a string into a float.

    Parameters
    ----------
    text : `str`
        Numeric text.
    dec_sep : `str` or `Iter[str]`
        Single or multiple decimal separators.
    """
    if is_number(text):
        return float(text)
    if isinstance(dec_sep, str):
        text = text.replace(dec_sep, ".")
    else:
        for ch in dec_sep:
            if ch in text:
                text = text.replace(ch, ".")
    return float(text)


###############################################################################
# Assumption Functions
###############################################################################


def assume_even_int(val):
    """Asserts that val is a scalar even integer."""
    val = int(val)
    return (val//2) * 2


def assume_odd_int(val):
    """Asserts that val is a scalar odd integer."""
    val = int(val)
    return (val//2) * 2 + 1


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
    elif isinstance(data, np.ndarray):
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
    elif isinstance(data, np.ndarray):
        return list(data.tolist())
    else:
        return [data]


def assume_numpy_array(data):
    """
    Asserts that data is a `numpy.ndarray` object.

    Parameters
    ----------
    data
        Input data to be checked for `numpy.ndarray`.

    Returns
    -------
    data : `np.ndarray`
        Returns a numpy array `data`.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
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
    obj : `object` or `dict`
        Object itself, keyword argument dictionary
        or single positional argument.
    cls_ : `class`
        Class of instance to be constructed.
    raise_exception : `Exception` or `None`
        Specifies the behaviour if object construction fails.
        `None`:
            Returns the input object.
        `Exception`:
            Raises the given exception.

    Returns
    -------
    obj : `cls_`
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
        try:
            return cls_(obj)
        except TypeError:
            if raise_exception is None:
                return obj
            else:
                raise raise_exception


def cv_bitfield(n):
    """
    Converts an integer to a list of booleans.

    Parameters
    ----------
    n : int
        Integer number representing the bit field.

    Returns
    -------
    bf : list(bool)
        Converted bitfield.
    """
    return [1 if digit == "1" else 0 for digit in bin(n)[2:]]


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
    func : `callable`
        Function to be mapped.
    *arg_dicts : `dict`
        Dictionaries whose values are mapped as arguments.
    kwarg_dicts : `dict(dict)`
        Dictionaries whose values are mapped as keyword arguments.
        The keys of kwarg_dicts correspond to the keywords.
    pre_args, post_args : `tuple`
        Arguments passed to func (before, after) dict
        values, i.e. `func(*pre_args, *dict_vals, *post_args)`.
    **kwargs
        Keyword arguments passed to `func`.

    Returns
    -------
    d : `dict`
        Mapped dictionary.

    Examples
    --------
    `map_dicts(func, d1, d2)` performs `new_dict[key] = func(d1, d2)` for all
    keys common to both `d1` and `d2`.
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


def rename_dict_keys(d, key_map, in_place=True):
    """
    Renames keys of a dictionary.

    Parameters
    ----------
    d : `dict`
        Dictionary to be renamed.
    key_map : `dict`
        Mapping between old key and new key.
    in_place : `bool`
        If `True`, changes the dictionary in place;
        if `False`, creates a copy of the dictionary.

    Returns
    -------
    d_new : `dict`
        Renamed dictionary.
    """
    d_new = d if in_place else d.copy()
    for k, v in key_map.items():
        if k in d_new:
            d_new[v] = d_new.pop(k)
    return d_new


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
    s : `str`
        String to be split.
    delim : `str` or `None`
        String delimiter. If None, string will not be split.
    strip : `str`
        Strip characters.

    Returns
    -------
    ls : `list(str)` or `str`
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
    s : `str`
        String to be split.

    Returns
    -------
    val : `int` or `float` or `str`
        Numerical value or unchanged string.
    unit : `str`
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
    s : `str`
        String from which to extract.
    regex : `str`
        Regular expression defining search function.
        Search findings should be enclosed in parentheses `()`.
    group : `int` or `list(int)` or `tuple(int)` or `np.ndarray(1, int)`
        Group index of search results.
        If list, returns corresponding list of search results.
    cv_func : `callable` or `None`
        Conversion function applied to search results (e.g. float).
    flags : `int`
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
    s : `str`
        String to be capitalized.

    Returns
    -------
    s_cap : `str`
        Capitalized string.
    """
    s_cap = re.sub(r"(\S)", lambda x: x.groups()[0].upper(), s, 1)
    return s_cap


REGEX_UPPERCASE_EUROPE = "[A-ZÀÁÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÖÙÚÛÜŒŸẞ]"
REGEX_LOWERCASE_EUROPE = "[a-zàáâäæçèéêëìíîïñòóôöùúûüœÿß]"


def cv_camel_to_snake_case(s, snake_char="_"):
    """
    Converts CamelCase to snake_case.

    Parameters
    ----------
    s : `str`
        String in CamelCase
    snake_char : `str`
        Snake case concatenation character.

    Returns
    -------
    s : `str`
        String in snake_case.
    """
    # Lower body of multi-upper-case strings
    s = re.sub(
        f"({REGEX_UPPERCASE_EUROPE})({REGEX_UPPERCASE_EUROPE}+)",
        lambda x: x.group(1) + x.group(2).lower(), s
    )
    # Add snake_char between lower and upper case characters
    s = re.sub(
        f"({REGEX_LOWERCASE_EUROPE}|\\d)({REGEX_UPPERCASE_EUROPE})",
        lambda x: x.group(1) + snake_char + x.group(2).lower(), s
    )
    return s.lower()


def char_range(start, stop=None, step=1):
    """
    Yield an alphabetic range of lowercase letters.

    Parameters
    ----------
    start : `chr`
        Start character.
        If stop is None: stop character (default start character: 'a').
    stop : `chr` or `int` or `None`
        If char: stop character (including stop).
        If int: number of steps (not including stop).
        If None: uses start as stop character
    step : `int`
        Character steps per repetition.

    Yields
    ------
    lr
        Letter range.
    """
    if stop is None:
        stop = start
        start = 'a'
    num_start, num_stop = ord(start.lower()), ord(start.lower())
    if isinstance(stop, int):
        num_stop += stop
    else:
        num_stop = ord(stop.lower()) + 1
    for ord_ in range(num_start, num_stop, step):
        yield chr(ord_)


def print_progress(
    count, total, subcount=None, subtotal=None, start_time=None,
    prefix="", fill="#", empty="·", bar_length=40, total_length=80, end=None
):
    if prefix != "":
        prefix += " "
    has_subprogress = subcount is not None and subtotal is not None
    progress = count / total
    if has_subprogress:
        progress = min(1, (count + subcount / subtotal) / (total))
    chars = int(round(progress * bar_length))
    progress_bar = "|{:s}{:s}| ".format(
        generate_fill_chars(chars, fill_char=fill),
        generate_fill_chars(bar_length - chars, fill_char=empty)
    )
    progress_str = "{:d}/{:d}".format(count, total)
    if has_subprogress:
        progress_str += " ({:d}/{:d})".format(subcount, subtotal)
    if start_time is not None:
        progress_str += ", {:.0f}s".format(time.time() - start_time)
    s = "\r" + prefix + progress_bar + progress_str
    diff_length = total_length - len(s)
    if diff_length > 0:
        s += generate_fill_chars(diff_length, fill_char=" ")
    if end is None:
        end = (count == total)
        if has_subprogress:
            end &= (subcount == subtotal)
    end = "\n" if end else ""
    print(s, end=end)


def iter_progress(it, **kwargs):
    """
    Returns an iterator and prints a progress bar for each iteration.

    Note that `it` must be sized (i.e. has a length).
    `**kwargs` are passed to :py:func:`print_progress`.
    """
    try:
        total = len(it)
    except TypeError:
        it = list(it)
        total = len(it)
    it = iter(it)
    if total == 0:
        return it
    start_time = time.time()
    counter = 0
    try:
        while True:
            print_progress(counter, total, start_time=start_time, **kwargs)
            yield next(it)
            counter += 1
    except StopIteration:
        return


###############################################################################
# Regular Expressions
###############################################################################


def get_regex_number(sgn=True, dec_sep="."):
    """
    Gets the regular expression for a signed decimal floating point number.

    Parameters
    ----------
    sgn : `bool`
        Whether to allow sign before number.
    dec_sep : `str`
        Decimal separator character. Multiple separators can be used by
        passing multiple characters, e.g. `".,"` for both point and comma as
        separators.
    """
    regex = r"[+-]?" if sgn else ""
    dec_sep = f"[{dec_sep}]?"
    regex += f"\\d*{dec_sep}\\d*"
    return regex


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


def get_combinations(ls, flatten=True, dtype=None):
    """
    Takes an iterable of iterables and returns a list with all possible
    mutual, ordered combinations.

    Parameters
    ----------
    ls : `Iter[Any]`
        Iterable from which combinations are constructed.
    flatten : `bool`
        Whether to flatten the items in `ls`.
        If `True`, the algorithm is faster but cannot work with e.g.
        items in `ls` that are multi-dimensional arrays.
    dtype : `None` or `callable`
        Combinations are given as numpy array.
        If `callable`, `dtype` is called on the returned numpy array.

    Returns
    -------
    comb : `np.ndarray` or `Any`
        Combinations list.

    Examples
    --------
    >>> ls = [(1, 2), (5, ), (7, 8)]
    >>> get_combinations(ls, dtype=list)
    [[1, 5, 7], [1, 5, 8], [2, 5, 7], [2, 5, 8]]
    """
    if flatten is True:
        comb = np.stack(np.meshgrid(*ls, indexing="ij"), axis=-1)
        comb = comb.reshape(-1, len(ls))
    else:
        if dtype is None:
            dtype = list
        comb = _gcrec([], ls)
    if dtype == list:
        if isinstance(comb, np.ndarray):
            comb = comb.tolist()
    elif callable(dtype):
        comb = dtype(dtype(x) for x in comb)
    else:
        comb = np.array(comb)
    return comb


###############################################################################
# Array Functions
###############################################################################


def get_numpy_dtype_str(dtype):
    """
    Gets the Python built-in type string corresponding to the given
    numpy dtype. If no matching built-in type is found, the numpy string
    representation of the dtype is returned.

    Parameters
    ----------
    dtype : `numpy.dtype`
        Numpy data type.

    Returns
    -------
    dtype_str : `str`
        String corresponding to numpy data type.
    """
    for dtype_str in ["bool", "int", "float", "complex", "object"]:
        if dtype == dtype_str:
            return dtype_str
    if "U" in str(dtype):
        return "str"
    return str(dtype)


def get_numpy_array_index(ar_or_ndim, dim, idx, default_idx=slice(None)):
    """
    Gets the multidimensional index to slice an array.

    Parameters
    ----------
    ar_or_ndim : `np.ndarray` or `int`
        Array or number of dimensions of targeted array.
    dim : `int` or `Iterable[int]`
        Index or indices of dimensions.
    idx : `int` or `slice` or `Iterable[int or slice]`
        Actual indices.
        If iterable is given, must be in the same order as `dim`.
    default_idx : `int` or `slice`
        Indices of dimensions not in `dim`.

    Returns
    -------
    slc : `tuple(int or slice)`
        Slicing indices.

    Examples
    --------
    >>> ar = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    >>> slc0 = get_numpy_array_index(ar, (0, 2), (0, 0))
    >>> np.all(ar[slc0] == ar[0, :, 0])
    True
    >>> slc1 = get_numpy_array_index(ar.ndim, (1, 2), (0, slice(0, 2)))
    >>> np.all(ar[slc1] == ar[:, 0, 0:2])
    True
    """
    ndim = ar_or_ndim if np.isscalar(ar_or_ndim) else ar_or_ndim.ndim
    dim, idx = assume_iter(dim), assume_iter(idx)
    slc = ndim * [default_idx]
    for i, d in enumerate(dim):
        slc[d] = idx[i]
    return tuple(slc)


def resize_numpy_array(ar, shape, fill_value=0, mode_keep="front", dtype=None):
    """
    Pads or shrinks a numpy array to a requested shape.

    Parameters
    ----------
    ar : `np.ndarray`
        Numpy array to be resized.
    shape : `tuple(int)`
        Target shape of array. Must have same dimension as `ar`.
    fill_value : `ar.dtype`
        Pad value if array is increased.
    mode_keep : `"front", "back", "center"`
        Keeps the elements of `ar` at the front/back/center of the
        resized array.
    dtype : `np.dtype` or `None`
        Numpy data type to cast to. If `None`, uses dtype of `ar`.

    Returns
    -------
    resized_ar : `np.ndarray`
        Resized numpy array.
    """
    # Find resizing index rectangle
    slice_new, slice_old = [], []
    for dim in range(ar.ndim):
        if mode_keep == "back":
            stop_new, stop_old = None, None
            if shape[dim] > ar.shape[dim]:
                start_new, start_old = shape[dim] - ar.shape[dim], None
            else:
                start_new, start_old = None, ar.shape[dim] - shape[dim]
        elif mode_keep == "center":
            center = shape[dim] // 2
            if shape[dim] > ar.shape[dim]:
                ldiff = ar.shape[dim] // 2
                rdiff = ar.shape[dim] - ldiff
                start_new, start_old = center - ldiff, None
                stop_new, stop_old = center + rdiff, None
            else:
                ldiff = shape[dim] // 2
                rdiff = shape[dim] - ldiff
                start_new, start_old = None, center - ldiff
                stop_new, stop_old = None, center + rdiff
        else:  # front
            start_new, start_old = None, None
            if shape[dim] > ar.shape[dim]:
                stop_new, stop_old = ar.shape[dim], None
            else:
                stop_new, stop_old = None, shape[dim]
        # Set slice
        slice_new.append(slice(start_new, stop_new))
        slice_old.append(slice(start_old, stop_old))
    # Resize
    resized_ar = np.full(
        shape, fill_value, dtype=ar.dtype if dtype is None else dtype
    )
    resized_ar[tuple(slice_new)] = ar[tuple(slice_old)]
    return resized_ar


def cv_array_points_to_bins(ar):
    """
    Converts a center point array to bin points.

    Dimensions: `(ndim0, ndim1, ...) -> (ndim0 + 1, ndim1 + 1, ...)`.

    Notes
    -----
    FIXME: Does not work for multiple dimensions.
    TODO: Does not apply proper dimensional averaging, only "diagonal"
    two-point average.
    """
    ar = np.array(ar)
    bins = np.zeros(np.array(ar.shape) + 1,
                    dtype=complex if ar.dtype == complex else float)
    slice_center = tuple(ar.ndim * [slice(1, -1)])
    slice_front = tuple(ar.ndim * [slice(None, -1)])
    slice_back = tuple(ar.ndim * [slice(1, None)])
    bins[slice_center] = (ar[slice_front] + ar[slice_back]) / 2
    print(bins)
    _gnai = get_numpy_array_index
    for dim in range(ar.ndim):
        bins[_gnai(bins, dim, 0, default_idx=slice(None, -1))] = (
            2 * ar[_gnai(ar, dim, 0)]
            - bins[_gnai(bins, dim, 1, default_idx=slice(None, -1))]
        )
        bins[_gnai(bins, dim, -1, default_idx=slice(1, None))] = (
            2 * ar[_gnai(ar, dim, -1)]
            - bins[_gnai(bins, dim, -2, default_idx=slice(1, None))]
        )
    return bins


def cv_index_center_to_rect(center, size):
    """
    Converts center-size indices to min-max indices.

    Parameters
    ----------
    center : `Iter[int]`
        Index center in each dimension.
    size : `Iter[int]` or `int`
        (Full) size of rectangle in each dimension.
        If `int`, the `size` is applied to each dimension.

    Returns
    -------
    rect : `Iter[(int, int)]`
        Index rectangle where the outer iterator corresponds to the dimension
        and the inner tuple corresponds to the slice `(min, max)`.
    """
    size = len(center) * [size] if np.isscalar(size) else size
    return tuple((max(0, c - size[i] // 2), c + (size[i] + 1) // 2)
                 for i, c in enumerate(center))


def cv_index_rect_to_slice(rect):
    """
    Converts min-max indices to slices.

    Parameters
    ----------
    rect : `Iter[(int, int) or slice]`
        Index rectangle where the outer iterator corresponds to the dimension
        and the inner tuple corresponds to the slice `(min, max)`.

    Returns
    -------
    slices : `Iter[slice]`
        Iterator of slices from index rectangle.
    """
    slices = []
    for item in rect:
        if isinstance(item, slice):
            slices.append(item)
        else:
            slices.append(slice(*item))
    return tuple(slices)


def cv_index_center_to_slice(center, size):
    """
    Shorthand for applying `cv_index_rect_to_slice(cv_index_center_to_rect())`.
    """
    return cv_index_rect_to_slice(cv_index_center_to_rect(center, size))


def transpose_array(ar):
    """
    Transposes a rectangular array.

    Parameters
    ----------
    ar : `numpy.ndarray` or `list(list)` or `tuple(tuple)`
        Rectangular array to be transposed.

    Returns
    -------
    ar : `numpy.ndarray` or `list(list)` or `tuple(tuple)`
        Transposed array.
    """
    if isinstance(ar, np.ndarray):
        return ar.T
    elif isinstance(ar, list):
        return list(map(list, zip(*ar)))
    else:
        return tuple(map(tuple, zip(*ar)))


def extract_index_nonunique_array(vec, min_count=2, rtol=1e-8):
    """
    Filters a vector for non-unique values and returns their indices.

    Parameters
    ----------
    vec : `np.ndarray(1)`
        Vector from which to extract indices of non-unique values.
    min_count : `int`
        Minimum multiplicity of values.
    rtol : `float`
        Relative tolerance defining the minimum deviation
        required to regard values as different.
        Only used for non-integer types.

    Returns
    -------
    idx : `iter(np.ndarray(1, int))`
        Iterable filter of lists of indices.
    """
    # Round precision errors (float->int->float)
    if vec.dtype == float:
        factor = 1 / np.max(np.abs(vec)) / rtol
        int_vec = (factor * vec).astype(int)
        vec = (int_vec / factor).astype(float)
    elif vec.dtype == complex:
        factor = 1 / np.max(np.abs(vec)) / rtol
        int_vec_re = (factor * np.real(vec)).astype(int)
        int_vec_im = (factor * np.imag(vec)).astype(int)
        vec = ((int_vec_re + 1j * int_vec_im) / factor).astype(complex)
    # Argsort vector to group same values
    idx_order = np.argsort(vec)
    vals, idx_start, counts = np.unique(
        vec[idx_order], return_index=True, return_counts=True
    )
    # Split argsort into index arrays of same vector value and filter unique
    idx = np.split(idx_order, idx_start[1:])
    idx = filter(lambda x: x.size >= min_count, idx)
    return idx


###############################################################################
# Coordinate Systems
###############################################################################


def cv_coord_spherical_to_cartesian(radial, polar, azimuthal):
    r"""
    Converts 3D spherical (standard parametrization) to cartesian coordinates.

    .. math:
        x = r \cdot \sin (\theta) \cos (\phi), \\
        y = r \cdot \sin (\theta) \sin (\phi), \\
        z = r \cdot \cos (\theta),

    where :math:`r \in [0, \infty]` is the radial, :math:`\theta \in [0, \pi]`
    is the polar, and :math:`\phi \in [0, 2 \pi]` is the azimuthal coordinate.
    Supports broadcasting.
    """
    x = radial * np.sin(polar) * np.cos(azimuthal)
    y = radial * np.sin(polar) * np.sin(azimuthal)
    z = radial * np.cos(polar)
    return np.array((x, y, z))


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


def make_getitem_func(arg):
    """
    Takes an object which implements `__getitem__` and accesses this method.

    Parameters
    ----------
    arg : `indexable`
        Object with __getitem__ method, e.g. dict or list.

    Returns
    -------
    func : `callable`
        Function signature: func(x)->y.
        Takes an argument and calls the getitem method. If the argument
        is invalid, the function returns identity.
    """
    if isinstance(arg, dict):
        return lambda x: arg[x] if x in arg.keys() else x
    elif isinstance(arg, list):
        return lambda x: arg[x] if (isinstance(x, int) and x < len(arg)) else x
    else:
        return lambda x: arg[x]


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
