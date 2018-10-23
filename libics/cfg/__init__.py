from . import default   # noqa
from . import env       # noqa
from . import err       # noqa

# System Imports
import abc
import copy


###############################################################################


class CfgBase(abc.ABC):

    """
    Base class for attribute based configuration classes.
    """

    def to_obj_dict(self):
        """
        Converts object dictionary to nested dictionary.

        Returns
        -------
        d : dict
            Nested configuration dictionary.
        """
        d = copy.deepcopy(self.__dict__)
        for attr, val in d.items():
            if isinstance(val, CfgBase):
                d[attr] = val.to_json()
        return d

    def from_obj_dict(self, d):
        """
        Converts nested configuration dictionary to object dictionary of
        current object.

        Parameters
        ----------
        d : dict
            Nested configuration dictionary.
        """
        for attr, val in d.items():
            if type(val) == dict:
                self.__dict__[attr] = val.from_obj_dict()
            else:
                self.__dict__[attr] = val

    @abc.abstractmethod
    def get_hl_cfg(self):
        """
        Gets the higher level configuration object.

        Returns
        -------
        obj : class derived from ProtocolCfgBase
            Highest level object.

        Notes
        ------
        Implement construction of higher level configuration class by defining
        a variable that classifies the next configuration level.
        This variable should be a class attribute of an enum-like class.
        This enum-class should have a dictionary MAP as attribute that maps
        the enum key to the class to be constructed. Using this map the higher
        level object can be dynamically created.
        """
