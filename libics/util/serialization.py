# System Imports
import copy
import io
import json
import numpy as np


###############################################################################
# Serialization by Dictionary
###############################################################################


class DictSerialization(object):

    """
    Base class for dictionary based serialization. Uses Python's built-in class
    dictionary to handle serialization.
    In case attributes have custom objects,
    pass a custom object (de-) serialization dictionary, so the algorithm knows
    for which attributes to call a custom (de-) serialization function. Using
    custom serialization implementations require deep copies of the class
    dictionary.
    """

    def __init__(self):
        self.obj_ser = {}    # Custom object serialization dictionary
        self.obj_deser = {}  # Custom object deserialization dictionary

    #   #### Setter ####################

    def add_obj_ser_func(self, obj_ser, obj_deser):
        r"""
        Adds object attribute name to (de-) serialization function dictionary.

        All object attributes are included in the serialized dictionary by
        default. If an attribute key is added to the (de-) serialization
        function dictionary, this function replaces the copy-and-assign
        (default) approach.

        This means that any structured attribute should implement a (de-)
        serialization function (explicitly as `(de-)serialize` or implicitly
        using a utility function). Built-in types can do not have to be
        added to the (de-) serialization function dictionary.

        Parameters
        ----------
        obj_ser : `dict`
            Object serialization function dictionary.
        obj_deser : `dict`
            Custom deserialization function dictionary.

        Notes
        -----
        (De-) serialization function dictionary entries have the following
        structure:

        key : `str`
            attribute_name
        value : `[func, *arg_func]` or `None`
            `[func, *arg_func]`: (De-) serialization is performed by
                value assigning where the value is retrieved by
                calling the function `func`.
                For serialization, `func(*arg_func)` is called, with
                `*arg_func` a list of non-parametrized getter
                functions that will be called.
                For deserialization, `func(serialized_value,
                *arg_func)`, with `serialized_value` the string to
                be deserialized and `*arg_func` a list of
                non-parametrized getter functions that will be
                called.
            `None`: Uses a default attribute (de-) serialization
                method which should be called serialize() and
                deserialize(serialized_value) respectively.
        """
        self.obj_ser.update(obj_ser)
        self.obj_deser.update(obj_deser)

    #   #### Calculators ###############

    def serialize(self):
        """
        Serializes all class attributes into a dictionary.

        Returns
        -------
        ser_dict : `dict`
            Class dictionary containing all attribute names as keys
            and attribute values as dictionary values.
        """
        # No custom object serialization
        if len(self.obj_ser) == 0:
            return self.__dict__
        # Custom object serialization
        else:
            ser_dict = copy.deepcopy(self.__dict__)
            for key in self.obj_ser:
                # Default serialize method implementation
                if self.obj_ser[key] is None:
                    ser_dict[key] = self.__dict__[key].serialize()
                # No serialize method implemented
                else:
                    args = [f() for f in self.obj_ser[key][1 : ]]  # noqa (E203)
                    ser_dict[key] = self.obj_ser[key][0](*args)
            return ser_dict

    def deserialize(self, ser_dict):
        """
        Deserializes all class attributes from a dictionary.

        Parameters
        ----------
        ser_dict : `dict`
            Imported dictionary containing class attributes as keys
            and values to be assigned.

        Notes
        -----
        Checks all passed dictionary items whether they have a corresponding
        key in the class dictionary. If this is the case, the passed value
        is assigned to the attribute.
        """
        for key in ser_dict.keys():
            if key in self.__dict__.keys():
                # Custom object deserialization
                if key in self.obj_deser.keys():
                    # Default deserialize method implementation
                    if self.obj_deser[key] is None:
                        self.__dict__[key].deserialize(ser_dict[key])
                    # No deserialize method implemented
                    else:
                        args = [f() for f in self.obj_deser[key][1 : ]]  # noqa (E203)
                        self.__dict__[key] = (
                            self.obj_deser[key][0](ser_dict[key], *args)
                        )
                # No custom object deserialization
                else:
                    self.__dict__[key] = ser_dict[key]


###############################################################################
# JSON Serialization Functions
###############################################################################


def numpy_ndarray_to_json(ndarray, codec="latin-1"):
    """
    Serializes an numpy.ndarray into binary form and encodes it in the
    specified codec. Returns the serialized json string including all
    information required to recover the numpy.ndarray object (pickle).

    Parameters
    ----------
    ndarray : `numpy.ndarray`
        numpy.ndarray to be serialized.
    codec : `str`
        Codec string (docs.python.org/3/library/codecs.html)

    Returns
    -------
    ser_string: `str`
        json string containing serialized numpy.ndarray
    """
    memfile = io.BytesIO()
    np.save(memfile, ndarray)
    memfile.seek(0)
    ser_string = json.dumps(memfile.read().decode(codec))
    return ser_string


def json_to_numpy_ndarray(ser_string, codec="latin-1"):
    """
    Deserializes a numpy.ndarray from a json string encoded in a
    specified codec. Returns the numpy.ndarray object.

    Parameters
    ----------
    ser_string : `str`
        json string containing serialized numpy.ndarray
    codec : `str`
        Codec string (docs.python.org/3/library/codecs.html)

    Returns
    -------
    ndarray: `numpy.ndarray`
        Deserialized numpy.ndarray.
    """
    memfile = io.BytesIO()
    memfile.write(json.loads(ser_string).encode(codec))
    memfile.seek(0)
    ndarray = np.load(memfile)
    return ndarray
