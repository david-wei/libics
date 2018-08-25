class FlaggedType:

    """
    Container for values with boolean flag.

    Provides a validity checker (range or subset based).
    Implements operators mathematical operators. On combinatory operation
    (e.g. `+`), flags are combined as `or` and conditions are dropped (`None`).

    Parameters
    ----------
    val
        Value to be stored.
    flag : bool
        Boolean flag.
    cond : tuple or list or None
        Value validity condition.
        tuple with length 2: range from min[0] to max[1].
        list: discrete set of allowed values.
        None: no validity check.
    """

    def __init__(self, val, flag=False, cond=None):
        self._val = None
        self._cond = None
        self.val = val
        self.flag = flag

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, v):
        if self._cond is not None:
            if type(self._cond) == tuple:
                if v < self._cond[0] or v > self._cond[1]:
                    raise ValueError("libics.util.types.FlaggedType.val")
            else:
                if v not in self._cond:
                    raise ValueError("libics.util.types.FlaggedType.val")
        self._val = v

    @property
    def cond(self):
        return self._cond

    @cond.setter
    def cond(self, c):
        if ((type(c) == tuple and len(c) == 2 and c[0] <= c[1])
                or type(c) == list):
            self._cond = c
        else:
            raise TypeError("libics.util.types.FlaggedType.cond")

    def invert(self):
        self.flag = not self.flag
        return self.flag

    def assign(self, other, diff_flag=False):
        """
        Copies the attributes of another `FlaggedType` object into the current
        object.

        Parameters
        ----------
        other : FlaggedType
            The object to be copied.
        diff_flag : bool
            Whether to copy the flag state or the differential flag
            state.
            `False`:
                Copies the flag state.
            `True`:
                Compares the values and sets the flag to
                `self.val != other.val`.
        """
        if diff_flag:
            self.flag = (self.val != other.val)
        else:
            self.flag = other.flag
        self.cond = other.cond
        self.val = other.val

    def __eq__(self, other):
        return self.val == other.val

    def __ne__(self, other):
        return self.val != other.val

    def __lt__(self, other):
        return self.val < other.val

    def __le__(self, other):
        return self.val <= other.val

    def __gt__(self, other):
        return self.val > other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __add__(self, other):
        return FlaggedType(
            self.val + other.val,
            flag=(self.flag or other.flag)
        )

    def __sub__(self, other):
        return FlaggedType(
            self.val - other.val,
            flag=(self.flag or other.flag)
        )

    def __mul__(self, other):
        return FlaggedType(
            self.val * other.val,
            flag=(self.flag or other.flag)
        )

    def __truediv__(self, other):
        return FlaggedType(
            self.val / other.val,
            flag=(self.flag or other.flag)
        )

    def __pow__(self, other):
        return FlaggedType(
            self.val**other.val,
            flag=(self.flag or other.flag)
        )

    def __neg__(self):
        return FlaggedType(-self.val, flag=self.flag, cond=self.cond)

    def __int__(self):
        return int(self.val)

    def __float__(self):
        return float(self.val)

    def __str__(self):
        return str(self.val)
