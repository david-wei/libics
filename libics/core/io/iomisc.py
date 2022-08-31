import json
import numpy as np


###############################################################################


class NumpyJsonEncoder(json.JSONEncoder):

    """
    JSON encoder class which converts numpy types to built-in types.

    Examples
    --------
    >>> ar = np.arange(5)
    >>> json.dumps(ar)
    TypeError: Object of type ndarray is not JSON serializable
    >>> json.dumps(ar, cls=NumpyJsonEncoder)
    '[0, 1, 2, 3, 4]'
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
