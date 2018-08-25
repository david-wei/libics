"""
Adds '/libics' to the system path.
"""


import inspect
import os
import sys


def _get_parent_dir(dir, level=1):
    """
    Gets the 'level's level parent directory of an absolute path 'dir'.

    The level specifies the directory level. Thus, if 'dir' is a file path,
    level 0 corresponds to the file's current directory.
    """
    if type(level) != int or type(dir) != str or level < 0:
        raise(TypeError)
    if os.path.isfile(dir):
        dir = os.path.dirname(dir)
    if level == 0:
        return dir
    else:
        dir = os.path.dirname(dir)
        level -= 1
        return _get_parent_dir(dir, level=level)


def _add_to_sys_path(path, index=None):
    """
    Adds 'path' to 'sys.path'.

    The 'index' specifies the path list position. If it is 'None', it will
    be appended. If 'path' is already in 'sys.path', 'sys.path' will not be
    changed, even if an index is specified.
    """
    if type(path) != str or (type(index) != int and index is not None):
        raise(TypeError)
    if path not in sys.path:
        if index is None:
            sys.path.append(path)
        else:
            if index < 0 or index >= len(sys.path):
                raise(ValueError)
            else:
                sys.path.insert(index, path)


_current_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())
))
_parent_dir = _get_parent_dir(_current_dir, level=2)
_add_to_sys_path(_parent_dir, index=None)
