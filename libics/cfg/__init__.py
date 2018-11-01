from . import default   # noqa
from . import env       # noqa
from . import err       # noqa

# System Imports
import abc
import collections
import copy


###############################################################################


class CfgBase(abc.ABC):

    """
    Base class for attribute based configuration classes.

    Parameters
    ----------
    group : str
        Group name this configuration belongs to.
    """

    def __init__(self, group="configuration"):
        self.group = group
        self._msg_queue = collections.deque()

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
            # FIXME: handle if attribute itself is a dictionary?!
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

    def _add_msg(self, msg):
        """
        Adds a message to a message queue (appends to last position) for
        subsequent processing.

        Parameters
        ----------
        msg : CfgMsg
            Message to be processed.
        """
        self._msg_queue.append(msg)

    def _pop_msg(self):
        """
        Gets a message from the message queue (from first position).

        Returns
        -------
        cfg_msg : CfgMsg
            First stored configuration message object.
        """
        msg = None
        try:
            msg = self._msg_queue.popleft()
        except(IndexError):
            pass
        return msg

    @property
    def val(self):
        """
        Unnecessary property function to obtain same behaviour as CfgItem.
        """
        return self


###############################################################################


class CFG_MSG_TYPE:

    IGNORE = 0
    WRITE = 1
    READ = 2
    VALIDATE = 3


class CfgMsg(object):

    def __init__(self, msg_type, value=None, callback=None):
        self.msg_type = msg_type
        self.value = value
        self._callback = callback

    def callback(self, *args, **kwargs):
        if self._callback is not None:
            return self._callback(*args, **kwargs)


class CfgItem(object):

    """
    Parameters
    ----------
    cfg : CfgBase
        Configuration class containing this item.
    name : str
        Name of the attribute represented by the instance.
    group : str
        Group name the attribute belongs to.
    val_check : None or list or tuple or type
        None:
            No value check.
        list:
            Checks that value is in list.
        tuple (length 2):
            Checks that value is within interval [min, max].
            If one boundary should not be checked, pass None.
        type:
            Checks type of value.
    """

    def __init__(self, cfg, name, group="general", val_check=None, val=None):
        self.cfg = cfg
        self.name = name
        self.group = group
        self._val = val
        self.val_check = None
        self.status = CFG_MSG_TYPE.IGNORE
        self.set_val_check(val_check=val_check)

    def set_val_check(self, val_check=None):
        if type(val_check) == tuple and len(val_check) == 2:
            if val_check[0] is None:
                self.val_check = lambda x: x >= val_check[0]
            elif val_check[1] is None:
                self.val_check = lambda x: x <= val_check[1]
            else:
                self.val_check = (
                    lambda x: x >= val_check[0] and x <= val_check[1]
                )
        if type(val_check) == list:
            self.val_check = lambda x: x in val_check
        if type(val_check) == type:
            self.val_check = lambda x: type(x) == val_check
        else:
            self.val_check = lambda x: True

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        if self.val_check(val):
            if self.status != CFG_MSG_TYPE.VALIDATE:
                self._val = val
        else:
            raise ValueError

    def read(self):
        """appends read msg"""
        self.status = CFG_MSG_TYPE.READ
        self.cfg._add_msg(CfgMsg(
            CFG_MSG_TYPE.READ,
            value=None,
            callback=(lambda x: setattr(self, "val", x))
        ))

    def write(self, val=None, validate=False):
        """appends write msg, potentially with validate read call"""
        if val is None:
            val = self.val
        elif not validate:
            self.val = val
        if validate:
            self.status = CFG_MSG_TYPE.VALIDATE
            self.cfg._add_msg(CfgMsg(
                CFG_MSG_TYPE.VALIDATE,
                value=self.val,
                callback=(lambda x: setattr(self, "val", x))
            ))
        else:
            self.status = CFG_MSG_TYPE.IGNORE
            self.cfg._add_msg(CfgMsg(
                CFG_MSG_TYPE.WRITE,
                value=self.val,
                callback=None
            ))

    def __lt__(self, other):
        if type(other) == CfgItem:
            other = other.val
        return self.val < other

    def __le__(self, other):
        if type(other) == CfgItem:
            other = other.val
        return self.val <= other

    def __eq__(self, other):
        if type(other) == CfgItem:
            other = other.val
        return self.val == other

    def __ne__(self, other):
        if type(other) == CfgItem:
            other = other.val
        return self.val != other

    def __ge__(self, other):
        if type(other) == CfgItem:
            other = other.val
        return self.val >= other

    def __gt__(self, other):
        if type(other) == CfgItem:
            other = other.val
        return self.val > other

    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self)


class CfgItemDesc:

    """
    Descriptor class for CfgBase attributes.

    Parameters
    ----------
    group : str
        Group name the attribute belongs to.
    val_check : None or list or tuple
        None:
            No value check.
        list:
            Checks that value is in list.
        tuple (length 2):
            Checks that value is within interval [min, max].
            If one boundary should not be checked, pass None.
    assert_write : bool
        Whether to read to validate a write command.
    """

    def __init__(self, group="general", val_check=None, assert_write=False):
        self.group = group
        self.val_check = val_check
        self.assert_write = assert_write

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if type(value) == CfgItem:
            instance.__dict__[self.name] = value
        else:
            instance.__dict__[self.name] = CfgItem(
                instance,
                self.name,
                group=self.group,
                val_check=self.val_check,
                val=value
            )
