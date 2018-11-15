from . import function          # noqa
from . import misc              # noqa
from . import path              # noqa
from . import pyqt              # noqa
from . import serialization     # noqa
from . import thread            # noqa
from . import types             # noqa


###############################################################################


class InheritMap:

    """
    Decorator class providing class inheritance mapping functionality.

    For decorated classes, a dictionary is created as class variable
    `map_name`. It maps `map_key` to the decorated class itself.
    Ensure that all classes along the inheritance tree are decorated.

    Parameters
    ----------
    map_name : str
        Name of the inherit map class variable.
        Default: "INH_MAP"
    map_key : any object suitable as dict key
        Inherit map dictionary key.
        None uses default: name of class.

    Notes
    -----
    Can be used to construct a child object from a base class and an
    inheritance map key, e.g.:
    >>> child_obj = parent_cls.INH_MAP[map_key]()
    """

    def __init__(self, map_name="INH_MAP", map_key=None):
        self.map_name = map_name
        self.map_key = map_key

    def __call__(self, cls):
        if self.map_key is None:
            self.map_key = cls.__name__
        if not hasattr(cls, self.map_name):
            setattr(cls, self.map_name, {self.map_key: cls})
        else:
            getattr(cls, self.map_name)[self.map_key] = cls
        return cls
