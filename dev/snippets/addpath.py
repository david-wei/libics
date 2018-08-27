"""
# Import Structure

## Library Structure

The library package consists of namespace-separating subpackages. These should
be maximally self-contained. Cross-imports should be only performed when base
classes and functions are excessively used, e.g. a data structure defined in
another subpackage.

## Subpackage Structure

The `__init__` module should only be accessed from the main package. Imports
within a subpackage should be directly referenced, i.e. each module does a
relative imports of all required modules. This also applies to deeper level
subpackages.

## Cross-Subpackage Imports

If cross-subpackage imports or imports from the parent folder are required, an
explicit module `addpath` should be placed within the subpackage. Modules
requiring cross-subpackage imports should perform a relative import of this
module. The required module or subpackage can then be absolutely imported as
`superpackage.subpackage` resp. `superpackage.subpackage.module`.
The code of this file is a template for the `addpath` module.

## Import Statement

An example import statement is given below. Preferred is a relative import
when the code is run in a package. If it is not run in a package, i.e. as main
or imported from the same subpackage, relative imports are invalid. Thus an
absolute import is performed.

try:
    from . import addpath
except(ImportError):
    import addpath
import superpackage.subpackage.module
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
